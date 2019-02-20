/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
// typedef struct {
//   float speeds[NSPEEDS];
// } t_speed;

typedef struct {
  float* speeds0;
  float* speeds1;
  float* speeds2;
  float* speeds3;
  float* speeds4;
  float* speeds5;
  float* speeds6;
  float* speeds7;
  float* speeds8;
} t_speeds;



/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* restrict paramfile, const char* restrict obstaclefile,
               t_param* restrict params, t_speeds** restrict cells_ptr, t_speeds** restrict tmp_cells_ptr,
               int** restrict obstacles_ptr, float** restrict av_vels_ptr);
/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells, int* restrict obstacles);
int accelerate_flow(const t_param params, t_speeds* restrict cells, int* restrict obstacles);
int propagate(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells);
// int rebound(const t_param params, t_speeds* cells, t_speeds* tmp_cells, int* obstacles);
float collision(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells, int* restrict obstacles);
int write_values(const t_param params, t_speeds* restrict cells, int* restrict obstacles, float* restrict av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* restrict params, t_speeds** restrict cells_ptr, t_speeds** restrict tmp_cells_ptr,
             int** restrict obstacles_ptr, float** restrict av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speeds* restrict cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speeds* restrict cells, int* restrict obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speeds* restrict cells, int* restrict obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
  char* paramfile = NULL;    /* name of the input parameter file */
  char* obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  t_speeds* cells = NULL;
  t_speeds* tmp_cells = NULL;

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  cells->speeds0 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);

  tmp_cells->speeds0 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);

 /* initialise densities */
  float w0 = params.density * 4.f / 9.f;
  float w1 = params.density      / 9.f;
  float w2 = params.density      / 36.f;

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* centre */
      cells->speeds0[ii + jj*params.nx] = w0;
      /* axis directions */
      cells->speeds1[ii + jj*params.nx] = w1;
      cells->speeds2[ii + jj*params.nx] = w1;
      cells->speeds3[ii + jj*params.nx] = w1;
      cells->speeds4[ii + jj*params.nx] = w1;
      /* diagonals */
      cells->speeds5[ii + jj*params.nx] = w2;
      cells->speeds6[ii + jj*params.nx] = w2;
      cells->speeds7[ii + jj*params.nx] = w2;
      cells->speeds8[ii + jj*params.nx] = w2;
    }
  }
  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  
  for (int tt = 0; tt < params.maxIters; tt++) {
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);
    t_speeds* holder = cells;
    cells = tmp_cells;
    tmp_cells = holder;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells, int* restrict obstacles) {
  accelerate_flow(params, cells, obstacles);
  //propagate(params, cells, tmp_cells);
  // rebound(params, cells, tmp_cells, obstacles);
  const float av = collision(params, cells, tmp_cells, obstacles);
  return av;
}

int accelerate_flow(const t_param params, t_speeds* restrict cells, int* restrict obstacles) {
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  // __assume_aligned(cells, 64);
  // __assume_aligned(cells->speeds0, 64);
  // __assume_aligned(cells->speeds1, 64);
  // __assume_aligned(cells->speeds2, 64);
  // __assume_aligned(cells->speeds3, 64);
  // __assume_aligned(cells->speeds4, 64);
  // __assume_aligned(cells->speeds5, 64);
  // __assume_aligned(cells->speeds6, 64);
  // __assume_aligned(cells->speeds7, 64);
  // __assume_aligned(cells->speeds8, 64);

  // __assume_aligned(obstacles, 64);
  #pragma omp simd
  for (int ii = 0; ii < params.nx; ii++) {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speeds6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speeds7[ii + jj*params.nx] - w2) > 0.f) {
      /* increase 'east-side' densities */
      cells->speeds1[ii + jj*params.nx] += w1;
      cells->speeds5[ii + jj*params.nx] += w2;
      cells->speeds8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds3[ii + jj*params.nx] -= w1;
      cells->speeds6[ii + jj*params.nx] -= w2;
      cells->speeds7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells) {

   __assume_aligned(cells, 64);
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speeds1, 64);
  __assume_aligned(tmp_cells->speeds2, 64);
  __assume_aligned(tmp_cells->speeds3, 64);
  __assume_aligned(tmp_cells->speeds4, 64);
  __assume_aligned(tmp_cells->speeds5, 64);
  __assume_aligned(tmp_cells->speeds6, 64);
  __assume_aligned(tmp_cells->speeds7, 64);
  __assume_aligned(tmp_cells->speeds8, 64);
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++) {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells->speeds0[ii + jj*params.nx] = cells->speeds0[ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx]; /* east */
      tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx]; /* north */
      tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx]; /* west */
      tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx]; /* south */
      tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

// int rebound(const t_param params, t_speeds* cells, t_speeds* tmp_cells, int* obstacles)
// {
//   /* loop over the cells in the grid */
//   for (int jj = 0; jj < params.ny; jj++)
//   {
//     for (int ii = 0; ii < params.nx; ii++)
//     {
//       /* if the cell contains an obstacle */
//       if (obstacles[jj*params.nx + ii])
//       {
//         /* called after propagate, so taking values from scratch space
//         ** mirroring, and writing into main grid */
//         cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
//         cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
//         cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
//         cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
//         cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
//         cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
//         cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
//         cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
//       }
//     }
//   }

//   return EXIT_SUCCESS;
// }

float collision(const t_param params, t_speeds* restrict cells, t_speeds* restrict tmp_cells, int* restrict obstacles) {
  const float c_sq = 3.f; /* square of speed of sound */
  const float halfc_sqsq = 4.5f;
  const float halfc_sq = 1.5f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);
  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speeds0, 64);
  __assume_aligned(tmp_cells->speeds1, 64);
  __assume_aligned(tmp_cells->speeds2, 64);
  __assume_aligned(tmp_cells->speeds3, 64);
  __assume_aligned(tmp_cells->speeds4, 64);
  __assume_aligned(tmp_cells->speeds5, 64);
  __assume_aligned(tmp_cells->speeds6, 64);
  __assume_aligned(tmp_cells->speeds7, 64);
  __assume_aligned(tmp_cells->speeds8, 64);
  __assume_aligned(obstacles, 64);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++) {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++) {

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      // /* propagate densities from neighbouring cells, following
      // ** appropriate directions of travel and writing into
      // ** scratch space grid */
      // tmp_cells->speeds0[ii + jj*params.nx] = cells->speeds0[ii + jj*params.nx]; /* central cell, no movement */
      // tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx]; /* east */
      // tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx]; /* north */
      // tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx]; /* west */
      // tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx]; /* south */
      // tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx]; /* north-east */
      // tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx]; /* north-west */
      // tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx]; /* south-west */
      // tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx]; /* south-east */

      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii]) {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds0[ii + jj*params.nx] = cells->speeds0[ii + jj*params.nx];
        tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx];
        tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx];
        tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx];
        tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx];
        tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx];
        tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx];
        tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx];
        tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx];
      }
      /* don't consider occupied cells */
      else {
        /* compute local density total */
        float local_density = 0.f;

        local_density += cells->speeds0[ii + jj*params.nx];
        local_density += cells->speeds1[x_w + jj*params.nx];
        local_density += cells->speeds2[ii + y_s*params.nx];
        local_density += cells->speeds3[x_e + jj*params.nx];
        local_density += cells->speeds4[ii + y_n*params.nx];
        local_density += cells->speeds5[x_w + y_s*params.nx];
        local_density += cells->speeds6[x_e + y_s*params.nx];
        local_density += cells->speeds7[x_e + y_n*params.nx];
        local_density += cells->speeds8[x_w + y_n*params.nx];
        
        /* compute x velocity component */
        const float u_x = (cells->speeds1[x_w + jj*params.nx]
                      + cells->speeds5[x_w + y_s*params.nx]
                      + cells->speeds8[x_w + y_n*params.nx]
                      - (cells->speeds3[x_e + jj*params.nx]
                         + cells->speeds6[x_e + y_s*params.nx]
                         + cells->speeds7[x_e + y_n*params.nx]))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (cells->speeds2[ii + y_s*params.nx]
                      + cells->speeds5[x_w + y_s*params.nx]
                      + cells->speeds6[x_e + y_s*params.nx]
                      - (cells->speeds4[ii + y_n*params.nx]
                         + cells->speeds7[x_e + y_n*params.nx]
                         + cells->speeds8[x_w + y_n*params.nx]))
                     / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;
        const float u_sqhalfc_sq = u_sq * halfc_sq;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq * halfc_sq);
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] * c_sq
                                         + (u[1] * u[1]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[2] = w1 * local_density * (1.f + u[2] * c_sq
                                         + (u[2] * u[2]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[3] = w1 * local_density * (1.f + u[3] * c_sq
                                         + (u[3] * u[3]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[4] = w1 * local_density * (1.f + u[4] * c_sq
                                         + (u[4] * u[4]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] * c_sq
                                         + (u[5] * u[5]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[6] = w2 * local_density * (1.f + u[6] * c_sq
                                         + (u[6] * u[6]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[7] = w2 * local_density * (1.f + u[7] * c_sq
                                         + (u[7] * u[7]) * halfc_sqsq
                                         - u_sqhalfc_sq);
        d_equ[8] = w2 * local_density * (1.f + u[8] * c_sq
                                         + (u[8] * u[8]) * halfc_sqsq
                                         - u_sqhalfc_sq);

        /* relaxation step */
        tmp_cells->speeds0[ii + jj*params.nx] = cells->speeds0[ii + jj*params.nx]
                                                + params.omega
                                                * (d_equ[0] - cells->speeds0[ii + jj*params.nx]);
        tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx]
                                                + params.omega
                                                * (d_equ[1] - cells->speeds1[x_w + jj*params.nx]);
        tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[2] - cells->speeds2[ii + y_s*params.nx]);
        tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx]
                                                + params.omega
                                                * (d_equ[3] - cells->speeds3[x_e + jj*params.nx]);
        tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[4] - cells->speeds4[ii + y_n*params.nx]);
        tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[5] - cells->speeds5[x_w + y_s*params.nx]);
        tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx]
                                                + params.omega
                                                * (d_equ[6] - cells->speeds6[x_e + y_s*params.nx]);
        tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[7] - cells->speeds7[x_e + y_n*params.nx]);
        tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx]
                                                + params.omega
                                                * (d_equ[8] - cells->speeds8[x_w + y_n*params.nx]);

        // /* local density total */
        // local_density = 0.f;

        // for (int kk = 0; kk < NSPEEDS; kk++)
        // {
        //   local_density += cells[ii + jj*params.nx].speeds[kk];
        // }

        // /* x-component of velocity */
        // u_x = (cells[ii + jj*params.nx].speeds[1]
        //               + cells[ii + jj*params.nx].speeds[5]
        //               + cells[ii + jj*params.nx].speeds[8]
        //               - (cells[ii + jj*params.nx].speeds[3]
        //                  + cells[ii + jj*params.nx].speeds[6]
        //                  + cells[ii + jj*params.nx].speeds[7]))
        //              / local_density;
        // /* compute y velocity component */
        // u_y = (cells[ii + jj*params.nx].speeds[2]
        //               + cells[ii + jj*params.nx].speeds[5]
        //               + cells[ii + jj*params.nx].speeds[6]
        //               - (cells[ii + jj*params.nx].speeds[4]
        //                  + cells[ii + jj*params.nx].speeds[7]
        //                  + cells[ii + jj*params.nx].speeds[8]))
        //              / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      
      }
    }
  }

  return  tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_speeds* restrict cells, int* restrict obstacles) {
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;
  __assume_aligned(cells, 64);
  __assume_aligned(cells->speeds0, 64);
  __assume_aligned(cells->speeds1, 64);
  __assume_aligned(cells->speeds2, 64);
  __assume_aligned(cells->speeds3, 64);
  __assume_aligned(cells->speeds4, 64);
  __assume_aligned(cells->speeds5, 64);
  __assume_aligned(cells->speeds6, 64);
  __assume_aligned(cells->speeds7, 64);
  __assume_aligned(cells->speeds8, 64);
  __assume_aligned(obstacles, 64);

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx]) {
        /* local density total */
        float local_density = 0.f;

        local_density += cells->speeds0[ii + jj*params.nx];
        local_density += cells->speeds1[ii + jj*params.nx];
        local_density += cells->speeds2[ii + jj*params.nx];
        local_density += cells->speeds3[ii + jj*params.nx];
        local_density += cells->speeds4[ii + jj*params.nx];
        local_density += cells->speeds5[ii + jj*params.nx];
        local_density += cells->speeds6[ii + jj*params.nx];
        local_density += cells->speeds7[ii + jj*params.nx];
        local_density += cells->speeds8[ii + jj*params.nx];

        /* x-component of velocity */
        float u_x = (cells->speeds1[ii + jj*params.nx]
                      + cells->speeds5[ii + jj*params.nx]
                      + cells->speeds8[ii + jj*params.nx]
                      - (cells->speeds3[ii + jj*params.nx]
                         + cells->speeds6[ii + jj*params.nx]
                         + cells->speeds7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds2[ii + jj*params.nx]
                      + cells->speeds5[ii + jj*params.nx]
                      + cells->speeds6[ii + jj*params.nx]
                      - (cells->speeds4[ii + jj*params.nx]
                         + cells->speeds7[ii + jj*params.nx]
                         + cells->speeds8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* restrict paramfile, const char* restrict obstaclefile,
               t_param* restrict params, t_speeds** restrict cells_ptr, t_speeds** restrict tmp_cells_ptr,
               int** restrict obstacles_ptr, float** restrict av_vels_ptr) {
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speeds*)_mm_malloc(sizeof(t_speeds), 64);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);


  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speeds*)_mm_malloc(sizeof(t_speeds) * (params->ny * params->nx), 64);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param* restrict params, t_speeds** restrict cells_ptr, t_speeds** restrict tmp_cells_ptr,
             int** restrict obstacles_ptr, float** restrict av_vels_ptr) {
  /*
  ** free up allocated memory
  */
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speeds* restrict cells, int* restrict obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speeds* restrict cells) {
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      total += cells->speeds0[ii + jj*params.nx];
      total += cells->speeds1[ii + jj*params.nx];
      total += cells->speeds2[ii + jj*params.nx];
      total += cells->speeds3[ii + jj*params.nx];
      total += cells->speeds4[ii + jj*params.nx];
      total += cells->speeds5[ii + jj*params.nx];
      total += cells->speeds6[ii + jj*params.nx];
      total += cells->speeds7[ii + jj*params.nx];
      total += cells->speeds8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speeds* restrict cells, int* restrict obstacles, float* restrict av_vels) {
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx]) {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      } else {
      /* no obstacle */
        local_density = 0.f;

        local_density += cells->speeds0[ii + jj*params.nx];
        local_density += cells->speeds1[ii + jj*params.nx];
        local_density += cells->speeds2[ii + jj*params.nx];
        local_density += cells->speeds3[ii + jj*params.nx];
        local_density += cells->speeds4[ii + jj*params.nx];
        local_density += cells->speeds5[ii + jj*params.nx];
        local_density += cells->speeds6[ii + jj*params.nx];
        local_density += cells->speeds7[ii + jj*params.nx];
        local_density += cells->speeds8[ii + jj*params.nx];
        /* compute x velocity component */
        u_x = (cells->speeds1[ii + jj*params.nx]
               + cells->speeds5[ii + jj*params.nx]
               + cells->speeds8[ii + jj*params.nx]
               - (cells->speeds3[ii + jj*params.nx]
                  + cells->speeds6[ii + jj*params.nx]
                  + cells->speeds7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds2[ii + jj*params.nx]
               + cells->speeds5[ii + jj*params.nx]
               + cells->speeds6[ii + jj*params.nx]
               - (cells->speeds4[ii + jj*params.nx]
                  + cells->speeds7[ii + jj*params.nx]
                  + cells->speeds8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}