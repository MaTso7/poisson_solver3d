// Manuel Tsolakis, 2019

// Solves the 3D Poisson equation for a given function f and Dirichlet boundary function g using Jacobi, Gauß-Seidel, SOR or CG
// f and g are specified by mesh.func_f and mesh.func_g in main() (e.g., f_sink and g_source)
// The result is written to file 'results' in binary, row-major form and can be read by, e.g., plot_poisson.m in MATLAB
// Additional parameters need to be specified via command line (order not important):
//
// alg=string 			which algorithm to use: "jacobi", "gs", "sor" or "cg"
// m=int 				the number of points along the first dimension
// n=int 				the number of points along the second dimension
// q=int 				the number of points along the third dimension
// h=double 			step size along all three dimensions
// threshold=double 	stopping criterion: J,GS,SOR: maximum local change from one iteration to the next, CG: length of residual vector
// x=double 			starting value for first dimension (including the boundary)
// y=double 			starting value for second dimension(including the boundary)
// z=double 			starting value for third dimension (including the boundary)

#include <stdio.h>
#include <string.h> //strcmp
#include <stdlib.h> //malloc, calloc, atof, atoi
#include <math.h>   //sqrt, sin, cos, fabs, cbrt
#include <mpi.h>

double f_sink(double x, double y, double z)
{
	return 2*( (1+x+z)*sin(x+y) - cos(x+y) );
}

double g_source(double x, double y, double z)
{
	return (1+x+z)*sin(x+y);
}


typedef struct
{//contains information purely about the MPI distribution
	int g_rank; 	  // global rank (MPI_COMM_WORLD)
	int size; 		  // # of processes
	MPI_Comm comm; 	  // grid communicator
	int dim_sizes[3]; // # of processes in each dimension
	int my_row; 	  // my row's coordinate
	int my_col; 	  // my column's coordinate
	int my_dep; 	  // my depth's coordinate
	int my_rank; 	  // my rank in the grid

	//neighbors along row,column and depth (minus 1 or plus 1)
	int rowm1, rowp1;
	int colm1, colp1;
	int depm1, depp1;
} GRID_INFO_T;

typedef struct
{//contains information of the actual mesh that is being solved over
	int m, n, q; // total # of elements in one dimension
	int local_m; // local # of elements row-wise, considers all points including boundaries
	int local_n; // local # of elements column-wise
	int local_q; // local # of elements depth-wise

	//pointers to arrays of all local_, but will consider only calculated points, not boundaries
	int *local_ms;
	int *local_ns;
	int *local_qs;

	double h; //step size
	double threshold; //stopping criterion

	//interval starting points, global and local
	double g_x_start; 
	double g_y_start;
	double g_z_start;

	double l_x_start; 
	double l_y_start;
	double l_z_start;

	double *u; //pointer to actual mesh
	double *h2f; //pointer to sink values*h^2
	double (*func_f)(double, double, double); //pointer to sink function
	double (*func_g)(double, double, double); //pointer to Dirichlet boundary function

	//MPI types for exchanging the boundaries
	MPI_Datatype send_types[3][2];
	MPI_Datatype recv_types[3][2];
} MESH_T;

enum Algorithm
{
	JACOBI,
	GS,
	SOR,
	CG,
	NONE
};


//forward declarations of functions
void read_input(MESH_T *mesh, enum Algorithm *alg, char **argv);

void setup_grid(GRID_INFO_T *grid);
void setup_mesh(MESH_T *mesh, GRID_INFO_T *grid);
void setup_MPI_Datatypes(MESH_T *mesh, GRID_INFO_T *grid);
void init_mesh(MESH_T *mesh, GRID_INFO_T *grid);
void init_mesh_cg(MESH_T *mesh, GRID_INFO_T *grid, double *rho, double *p);

void jacobi(MESH_T *mesh, GRID_INFO_T *grid);
void gauss_seidel(MESH_T *mesh, GRID_INFO_T *grid);
void gs_sor(MESH_T *mesh, GRID_INFO_T *grid);
void conjugate_gradient(MESH_T *mesh, GRID_INFO_T *grid);

void write_to_file(MESH_T *mesh, GRID_INFO_T *grid);



void setup_grid(GRID_INFO_T *grid)
{//create an MPI Cartesian communicator
	//global stuff
	MPI_Comm_rank( MPI_COMM_WORLD, &(*grid).g_rank );
	MPI_Comm_size( MPI_COMM_WORLD, &(*grid).size );
	if((*grid).g_rank == 0)
		printf("Using %d processes.\n", (*grid).size);


	//create Cartesian grid
	const int dimensions = 3;
	MPI_Dims_create( (*grid).size, dimensions, (*grid).dim_sizes );
	int wrap_around[3] = {0}; // no wrap around, i.e., no periodic boundaries
	int reorder = 1;

	MPI_Cart_create(MPI_COMM_WORLD, dimensions, (*grid).dim_sizes, wrap_around, reorder, &(*grid).comm);
	

	//get rank and coordinates
	MPI_Comm_rank( (*grid).comm, &(*grid).my_rank );
	
	int coordinates[dimensions];
	MPI_Cart_coords( (*grid).comm, (*grid).my_rank, dimensions, coordinates );
	(*grid).my_row = coordinates[0];
	(*grid).my_col = coordinates[1];
	(*grid).my_dep = coordinates[2];


	//get ranks of neighbors
	//rows
	if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
	{
		coordinates[0]++; 
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).rowp1 );
	}
	else
		(*grid).rowp1 = -1;

	if( (*grid).my_row != 0 )
	{
		coordinates[0] = (*grid).my_row - 1;
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).rowm1 );
	}
	else
		(*grid).rowm1 = -1;

	coordinates[0] = (*grid).my_row;

	//columns
	if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
	{
		coordinates[1]++; 
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).colp1 );
	}
	else
		(*grid).colp1 = -1;

	if( (*grid).my_col != 0 )
	{
		coordinates[1] = (*grid).my_col - 1;
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).colm1 );
	}
	else
		(*grid).colm1 = -1;

	coordinates[1] = (*grid).my_col;

	//depths
	if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
	{
		coordinates[2]++;
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).depp1 );
	}
	else
		(*grid).depp1 = -1;

	if( (*grid).my_dep != 0 )
	{
		coordinates[2]  = (*grid).my_dep - 1;
		MPI_Cart_rank( (*grid).comm, coordinates, &(*grid).depm1 );
	}
	else
		(*grid).depm1 = -1;
	//coordinates[2] = (*grid).my_dep; //unnecessary

	return;
}


void read_input(MESH_T *mesh, enum Algorithm *alg, char **argv)
{//read necessary values from command line arguments
	for(int i = 1; i < 10; i++)
	{
		char *current = argv[i];
		if( strncmp("alg=", current, strlen("alg=")) == 0 )
		{
			current += strlen("alg=");
			if( strncmp("jacobi", current, strlen("jacobi")) == 0 )
				*alg = JACOBI;
			else if( strncmp("gs", current, strlen("gs")) == 0 )
				*alg = GS;
			else if( strncmp("sor", current, strlen("sor")) == 0 )
				*alg = SOR;
			else if( strncmp("cg", current, strlen("cg")) == 0 )
				*alg = CG;
			else
			{
				fprintf(stderr, "Command line: Unrecognized algorithm.\n");
				MPI_Abort(MPI_COMM_WORLD, 4);
			}
			continue;
		}
		if( strncmp("n=", current, strlen("n=")) == 0 )
		{
			current += strlen("n=");
			(*mesh).n = atoi(current);
			continue;
		}
		if( strncmp("m=", current, strlen("m=")) == 0 )
		{
			current += strlen("m=");
			(*mesh).m = atoi(current);
			continue;
		}
		if( strncmp("q=", current, strlen("q=")) == 0 )
		{
			current += strlen("q=");
			(*mesh).q = atoi(current);
			continue;
		}
		if( strncmp("h=", current, strlen("h=")) == 0 )
		{
			current += strlen("h=");
			(*mesh).h = atof(current);
			continue;
		}
		if( strncmp("threshold=", current, strlen("threshold=")) == 0 )
		{
			current += strlen("threshold=");
			(*mesh).threshold = atof(current);
			continue;
		}
		if( strncmp("x=", current, strlen("x=")) == 0 )
		{
			current += strlen("x=");
			(*mesh).g_x_start = atof(current);
			continue;
		}
		if( strncmp("y=", current, strlen("y=")) == 0 )
		{
			current += strlen("y=");
			(*mesh).g_y_start = atof(current);
			continue;
		}
		if( strncmp("z=", current, strlen("z=")) == 0 )
		{
			current += strlen("z=");
			(*mesh).g_z_start = atof(current);
			continue;
		}
	}
	return;
}


void setup_mesh(MESH_T *mesh, GRID_INFO_T *grid)
{//decompose the given volume into smaller ones distributed among all processes
	int m = (*mesh).m;
	int n = (*mesh).n;
	int q = (*mesh).q;

	//checking for correct usage
	if( ( m*n*q < (*grid).size * 8 ) && ( (*grid).g_rank == 0 ) )
	{
		fprintf(stderr, "Each process should have at least 8 elements.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//calculate sizes (for all processes) and allocate
	(*mesh).local_ms = malloc(sizeof *(*mesh).local_ms * (*grid).dim_sizes[0]);
	for(int i = 0; i < (*grid).dim_sizes[0]; i++)
		(*mesh).local_ms[i] = m/(*grid).dim_sizes[0] + (i < m%(*grid).dim_sizes[0] ? 1 : 0);

	(*mesh).local_ns = malloc(sizeof *(*mesh).local_ns * (*grid).dim_sizes[1]);
	for(int i = 0; i < (*grid).dim_sizes[1]; i++)
		(*mesh).local_ns[i] = n/(*grid).dim_sizes[1] + (i < n%(*grid).dim_sizes[1] ? 1 : 0);

	(*mesh).local_qs = malloc(sizeof *(*mesh).local_qs * (*grid).dim_sizes[2]);
	for(int i = 0; i < (*grid).dim_sizes[2]; i++)
		(*mesh).local_qs[i] = q/(*grid).dim_sizes[2] + (i < q%(*grid).dim_sizes[2] ? 1 : 0);

	(*mesh).local_m = (*mesh).local_ms[(*grid).my_row] + 2; //+2 for included boundary points
	(*mesh).local_n = (*mesh).local_ns[(*grid).my_col] + 2;
	(*mesh).local_q = (*mesh).local_qs[(*grid).my_dep] + 2;

	(*mesh).u = calloc( (*mesh).local_m * (*mesh).local_n * (*mesh).local_q, sizeof *(*mesh).u );
	(*mesh).h2f = malloc( sizeof *(*mesh).h2f * (*mesh).local_m * (*mesh).local_n * (*mesh).local_q );

	if((*mesh).u == NULL || (*mesh).h2f == NULL)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		MPI_Abort(MPI_COMM_WORLD,7);
	}

	//calculate individual starting points
	double h = (*mesh).h;

	(*mesh).l_x_start = (*mesh).g_x_start;
	(*mesh).l_y_start = (*mesh).g_y_start;
	(*mesh).l_z_start = (*mesh).g_z_start;

	for(int i = 0; i < (*grid).my_row; i++)
	{
		(*mesh).l_x_start += h*(*mesh).local_ms[i];
	}

	for(int j = 0; j < (*grid).my_col; j++)
	{
		(*mesh).l_y_start += h*(*mesh).local_ns[j];
	}

	for(int k = 0; k < (*grid).my_dep; k++)
	{
		(*mesh).l_z_start += h*(*mesh).local_qs[k];
	}


	setup_MPI_Datatypes(mesh, grid);

	return;
}

void setup_MPI_Datatypes(MESH_T *mesh, GRID_INFO_T *grid)
{//sets up the MPI_Datatype s that are used to exchange the boundaries, created as subarrays
	int sizes[3] = { (*mesh).local_m, (*mesh).local_n, (*mesh).local_q };

	// as boundary points belong to other processors, we have offsets of 1 in sending and receiving
	int subsizes[3] = { 1, (*mesh).local_n-2, (*mesh).local_q-2 };
	if( (*grid).my_row != 0 )
	{	//rowm1: send to "left", receive from "left"
		int send_starts[3] = { 1, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[0][0]);
		MPI_Type_commit(&(*mesh).send_types[0][0]);

		int recv_starts[3] = { 0, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[0][1]);
		MPI_Type_commit(&(*mesh).recv_types[0][1]);
	}

	if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
	{	//rowp1: send to "right", receive from "right"
		int send_starts[3] = { (*mesh).local_m-2, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[0][1]);
		MPI_Type_commit(&(*mesh).send_types[0][1]);

		int recv_starts[3] = { (*mesh).local_m-1, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[0][0]);
		MPI_Type_commit(&(*mesh).recv_types[0][0]);
	}


	int subsizes2[3] = { (*mesh).local_m-2, 1, (*mesh).local_q-2 };
	if( (*grid).my_col != 0 )
	{	//colm1
		int send_starts[3] = { 1, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes2, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[1][0]);
		MPI_Type_commit(&(*mesh).send_types[1][0]);

		int recv_starts[3] = { 1, 0, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes2, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[1][1]);
		MPI_Type_commit(&(*mesh).recv_types[1][1]);
	}

	if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
	{	//colp1
		int send_starts[3] = { 1, (*mesh).local_n-2, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes2, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[1][1]);
		MPI_Type_commit(&(*mesh).send_types[1][1]);

		int recv_starts[3] = { 1, (*mesh).local_n-1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes2, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[1][0]);
		MPI_Type_commit(&(*mesh).recv_types[1][0]);
	}


	int subsizes3[3] = { (*mesh).local_m-2, (*mesh).local_n-2, 1 };
	if( (*grid).my_dep != 0 )
	{	//depm1
		int send_starts[3] = { 1, 1, 1 };
		MPI_Type_create_subarray(3, sizes, subsizes3, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[2][0]);
		MPI_Type_commit(&(*mesh).send_types[2][0]);

		int recv_starts[3] = { 1, 1, 0 };
		MPI_Type_create_subarray(3, sizes, subsizes3, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[2][1]);
		MPI_Type_commit(&(*mesh).recv_types[2][1]);
	}

	if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
	{	//depp1
		int send_starts[3] = { 1, 1, (*mesh).local_q-2 };
		MPI_Type_create_subarray(3, sizes, subsizes3, send_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).send_types[2][1]);
		MPI_Type_commit(&(*mesh).send_types[2][1]);

		int recv_starts[3] = { 1, 1, (*mesh).local_q-1 };
		MPI_Type_create_subarray(3, sizes, subsizes3, recv_starts, MPI_ORDER_C, MPI_DOUBLE, &(*mesh).recv_types[2][0]);
		MPI_Type_commit(&(*mesh).recv_types[2][0]);
	}

	return;
}

void init_mesh(MESH_T *mesh, GRID_INFO_T *grid)
{//initialize the mesh (boundaries and h2f need to be computed only once)
	//initialize h2f
	double h = (*mesh).h;
	for(int i = 1; i < (*mesh).local_m - 1; i++)
	{
		for(int j = 1; j < (*mesh).local_n - 1; j++)
		{
			for(int k = 1; k < (*mesh).local_q - 1; k++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + j*(*mesh).local_q + k;
				(*mesh).h2f[index] = h*h * (*mesh).func_f( (*mesh).l_x_start+i*h, (*mesh).l_y_start+j*h, (*mesh).l_z_start+k*h );
			}
		}
	}


	//initialize u=g

	if( (*grid).my_row == 0 )
	{ //i = 0
		for(int j = 0; j < (*mesh).local_n; j++)
		{
			for(int k = 0; k < (*mesh).local_q; k++)
			{
				int index = 0 + j*(*mesh).local_q + k;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).g_x_start, (*mesh).l_y_start+j*h, (*mesh).l_z_start+k*h );
			}
		}
	}

	if( (*grid).my_row == (*grid).dim_sizes[0]-1 )
	{
		int i = (*mesh).local_m - 1;
		for(int j = 0; j < (*mesh).local_n; j++)
		{
			for(int k = 0; k < (*mesh).local_q; k++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + j*(*mesh).local_q + k;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).l_x_start+i*h, (*mesh).l_y_start+j*h, (*mesh).l_z_start+k*h );
			}
		}
	}

	if( (*grid).my_col == 0 )
	{ //j = 0
		for(int i = 0; i < (*mesh).local_m; i++)
		{
			for(int k = 0; k < (*mesh).local_q; k++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + 0 + k;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).l_x_start+i*h, (*mesh).g_y_start, (*mesh).l_z_start+k*h );
			}
		}
	}

	if( (*grid).my_col == (*grid).dim_sizes[1]-1 )
	{
		int j = (*mesh).local_n - 1;
		for(int i = 0; i < (*mesh).local_m; i++)
		{
			for(int k = 0; k < (*mesh).local_q; k++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + j*(*mesh).local_q + k;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).l_x_start+i*h, (*mesh).l_y_start+j*h, (*mesh).l_z_start+k*h );
			}
		}
	}

	if( (*grid).my_dep == 0 )
	{ //k = 0
		for(int i = 0; i < (*mesh).local_m; i++)
		{
			for(int j = 0; j < (*mesh).local_n; j++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + j*(*mesh).local_q + 0;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).l_x_start+i*h, (*mesh).l_y_start+j*h, (*mesh).g_z_start );
			}
		}
	}

	if( (*grid).my_dep == (*grid).dim_sizes[2]-1 )
	{
		int k = (*mesh).local_q - 1;
		for(int i = 0; i < (*mesh).local_m; i++)
		{
			for(int j = 0; j < (*mesh).local_n; j++)
			{
				int index = i*(*mesh).local_n*(*mesh).local_q + j*(*mesh).local_q + k;
				(*mesh).u[index] = (*mesh).func_g( (*mesh).l_x_start+i*h, (*mesh).l_y_start+j*h, (*mesh).l_z_start+k*h );
			}
		}
	}

	return;
}


void jacobi(MESH_T *mesh, GRID_INFO_T *grid)
{
	//create a second mesh to be able to store old AND new values and initialize it
	double *u_temp = malloc(sizeof *u_temp * (*mesh).local_m * (*mesh).local_n * (*mesh).local_q);
	if(u_temp == NULL)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		MPI_Abort(MPI_COMM_WORLD,6);
	}
	double *swap_pointer = (*mesh).u;
	(*mesh).u = u_temp;
	init_mesh(mesh, grid);
	(*mesh).u = swap_pointer;

	if((*grid).my_rank == 0)
		printf("Starting Jacobi\nMaximal change after the ith iteration:\n");
	double max_diff;
	do
	{
		max_diff = 0;
		//start actual Jacobi
		for(int i = 1; i < (*mesh).local_m - 1; i++)
		{
			for(int j = 1; j < (*mesh).local_n - 1; j++)
			{
				for(int k = 1; k < (*mesh).local_q - 1; k++)
				{
					int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
					int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
					int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
					int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

					double diff = (*mesh).u[index];
					u_temp[index] = ( 	(*mesh).h2f[index] 
						+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
						+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
						+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
					diff -= u_temp[index];
					if( fabs(diff) > max_diff )
						max_diff = fabs(diff);
				}
			}
		}

		//now that the iteration is finished, replace old values by the new ones
		swap_pointer = (*mesh).u;
		(*mesh).u = u_temp;
		u_temp = swap_pointer;


		//start point exchange
		MPI_Request requests[3*2*2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
		
		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}

		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}

		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}

		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}


		//look for maximum change that happened in the current iteration
		MPI_Allreduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, (*grid).comm);


		//print status occasionally and at the end
		static int count = 0;
		count++;
		if((*grid).my_rank == 0 && (count == 1 || count%200 == 0))
			printf("%d: %e\n", count, max_diff);
		else if((*grid).my_rank == 0 && max_diff <= (*mesh).threshold)
			printf("%d: %e\n", count, max_diff);


		//make sure all necessary communication is done by now
		MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

	}while(max_diff > (*mesh).threshold);

	free(u_temp);

	return;
}


void gauss_seidel(MESH_T *mesh, GRID_INFO_T *grid)
{
	//calculate starting corner to know whether the first point is even or odd / red or black
	int start_even_odd = 0;
	for(int i = 0; i < (*grid).my_row; i++)
		start_even_odd += (*mesh).local_ms[i];
	for(int j = 0; j < (*grid).my_col; j++)
		start_even_odd += (*mesh).local_ns[j];
	for(int k = 0; k < (*grid).my_dep; k++)
		start_even_odd += (*mesh).local_qs[k];


	if((*grid).my_rank == 0)
		printf("Starting Gauß-Seidel\nMaximal change after the ith iteration:\n");
	double max_diff;
	do
	{
		max_diff = 0;
		//actually start GS

		//start with red / even points
		if(start_even_odd%2)
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = ( 	(*mesh).h2f[index] 
												+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
												+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
												+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}
		else //note the change in k
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j-1)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = ( 	(*mesh).h2f[index] 
												+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
												+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
												+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}


		//start point exchange
		MPI_Request requests[3*2*2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
		
		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}

		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}

		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}

		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}

		MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);


		//now do black / odd points
		if(start_even_odd%2)
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j-1)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = ( 	(*mesh).h2f[index] 
												+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
												+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
												+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}
		else
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = ( 	(*mesh).h2f[index] 
												+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
												+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
												+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}


		//start point exchange
		for(int i = 0; i < 3*2*2; i++)
			requests[i] = MPI_REQUEST_NULL;

		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}

		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}

		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}

		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}


		//look for maximum change in the current iteration
		MPI_Allreduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, (*grid).comm);


		//print status occasionally and at the end
		static int count = 0;
		count++;
		if((*grid).my_rank == 0 && (count == 1 || count%200 == 0))
			printf("%d: %e\n", count, max_diff);
		else if((*grid).my_rank == 0 && max_diff <= (*mesh).threshold)
			printf("%d: %e\n", count, max_diff);


		//make sure all necessary communication is done by now
		MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

	}while(max_diff > (*mesh).threshold);

	return;
}


void gs_sor(MESH_T *mesh, GRID_INFO_T *grid)
{//Gauß-Seidel with successive over-relaxation

	//calculate the relaxation factor omega
	double n_est = cbrt((*mesh).m * (*mesh).n * (*mesh).q);
	double rhosor = 1 - 4*3.14159265358979323846/(n_est+1);
	double omega = 2/(1+sqrt(1-rhosor*rhosor));
	//double omega = 1.5;

	//calculate starting corner to know whether the first point is even or odd / red or black
	int start_even_odd = 0;
	for(int i = 0; i < (*grid).my_row; i++)
		start_even_odd += (*mesh).local_ms[i];
	for(int j = 0; j < (*grid).my_col; j++)
		start_even_odd += (*mesh).local_ns[j];
	for(int k = 0; k < (*grid).my_dep; k++)
		start_even_odd += (*mesh).local_qs[k];


	if((*grid).my_rank == 0)
		printf("Starting SOR\nMaximal change after the ith iteration using omega=%1.3f:\n",omega);
	double max_diff;
	do
	{
		max_diff = 0;
		//actually start SOR

		//start with red / even points
		if(start_even_odd%2)
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = (1-omega)*(*mesh).u[index] 
											+ omega*( 	(*mesh).h2f[index] 
													+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
													+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
													+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}
		else
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j-1)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = (1-omega)*(*mesh).u[index] 
											+ omega*( 	(*mesh).h2f[index] 
													+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
													+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
													+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}


		//start point exchange
		MPI_Request requests[3*2*2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
		
		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}

		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}

		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}

		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}


		//now do black / odd points
		if(start_even_odd%2)
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j-1)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = (1-omega)*(*mesh).u[index] 
											+ omega*( 	(*mesh).h2f[index] 
													+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
													+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
													+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}
		else
		{
			for(int i = 1; i < (*mesh).local_m - 1; i ++)
			{
				for(int j = 1; j < (*mesh).local_n - 1; j++)
				{
					for(int k = 1+(i+j)%2; k < (*mesh).local_q - 1; k+=2)
					{
						int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
						int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
						int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
						int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
						int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

						double diff = (*mesh).u[index];
						(*mesh).u[index] = (1-omega)*(*mesh).u[index] 
											+ omega*( 	(*mesh).h2f[index] 
													+ (*mesh).u[ind_ip] + (*mesh).u[ind_im]
													+ (*mesh).u[ind_jp] + (*mesh).u[ind_jm] 
													+ (*mesh).u[ind_kp] + (*mesh).u[ind_km] ) / 6;
						diff -= (*mesh).u[index];
						if( fabs(diff) > max_diff )
							max_diff = fabs(diff);
					}
				}
			}
		}


		//start point exchange
		for(int i = 0; i < 3*2*2; i++)
			requests[i] = MPI_REQUEST_NULL;

		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}

		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}

		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}

		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( (*mesh).u, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( (*mesh).u, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}


		//look for maximum change in the current iteration
		MPI_Allreduce(MPI_IN_PLACE, &max_diff, 1, MPI_DOUBLE, MPI_MAX, (*grid).comm);


		//print status occasionally and at the end
		static int count = 0;
		count++;
		if((*grid).my_rank == 0 && (count == 1 || count%200 == 0))
			printf("%d: %e\n", count, max_diff);
		else if((*grid).my_rank == 0 && max_diff <= (*mesh).threshold)
			printf("%d: %e\n", count, max_diff);


		//make sure all necessary communication is done by now
		MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

	}while(max_diff > (*mesh).threshold);

	return;
}


void init_mesh_cg(MESH_T *mesh, GRID_INFO_T *grid, double *rho, double *p)
{
	double *x = (*mesh).u;
	double *r = (*mesh).h2f;

	//add the boundary points to the right hand side b which is also r0
	//since x0 = u0 = 0, all values can be specified; non-boundary points will be 0
	for(int i = 1; i < (*mesh).local_m - 1; i++)
	{
		for(int j = 1; j < (*mesh).local_n - 1; j++)
		{
			for(int k = 1; k < (*mesh).local_q - 1; k++)
			{
				int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
				int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
				int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
				int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
				int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
				int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
				int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

				r[index] +=   x[ind_ip] + x[ind_im]
							+ x[ind_jp] + x[ind_jm]
							+ x[ind_kp] + x[ind_km];
			}
		}
	}

	for( int i = 1; i < (*mesh).local_m - 1; i++)
	{
		for( int j = 1; j < (*mesh).local_n - 1; j++)
		{
			for( int k = 1; k < (*mesh).local_q - 1; k++)
			{
				int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
				p[index] = r[index];
				*rho += r[index]*r[index];
			}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, rho, 1, MPI_DOUBLE, MPI_SUM, (*grid).comm);


	//p needs to be initialized at inner boundary points as well
	//this could also be done via actual calculations instead of communication (and probably would be faster but it's only once)
	MPI_Request requests[3*2*2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
									MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
									MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
	//exchange row-wise
	if( (*grid).my_row != 0 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
		MPI_Irecv( p, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
	}
	if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
		MPI_Irecv( p, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
	}

	//exchange column-wise
	if( (*grid).my_col != 0 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
		MPI_Irecv( p, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
	}
	if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
		MPI_Irecv( p, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
	}

	//exchange depth-wise
	if( (*grid).my_dep != 0 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
		MPI_Irecv( p, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
	}
	if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
	{
		MPI_Isend( p, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
		MPI_Irecv( p, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
	}

	MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

	return;
}

void conjugate_gradient(MESH_T *mesh, GRID_INFO_T *grid)
{
	//renaming of existing memory and new allocation
	double *x = (*mesh).u;
	double *r = (*mesh).h2f;

	double alpha;
	double *s = malloc( sizeof *s * (*mesh).local_m * (*mesh).local_n * (*mesh).local_q );
	double *p = calloc( (*mesh).local_m * (*mesh).local_n * (*mesh).local_q, sizeof *p );
	if(p == NULL || s == NULL)
	{
		fprintf(stderr, "Memory allocation failed.\n");
		MPI_Abort(MPI_COMM_WORLD,6);
	}
	double rho[2] = {0};

	init_mesh_cg(mesh, grid, &rho[0], p);


	if((*grid).g_rank == 0)
		printf("Starting CG\nLength of residual vector after ith iteration:\n");
	//start CG
	do
	{
		alpha = 0;
		for(int i = 1; i < (*mesh).local_m - 1; i++)
		{
			for(int j = 1; j < (*mesh).local_n - 1; j++)
			{
				for(int k = 1; k < (*mesh).local_q - 1; k++)
				{
					int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_ip= (i+1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_im= (i-1)*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;
					int ind_jp= 	i*(*mesh).local_n*(*mesh).local_q + (j+1)*(*mesh).local_q + k;
					int ind_jm= 	i*(*mesh).local_n*(*mesh).local_q + (j-1)*(*mesh).local_q + k;
					int ind_kp= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k+1;
					int ind_km= 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k-1;

					//again, boundary points are set to 0 (by calloc) because they are already included in b
					s[index] = 6*p[index] 
								- p[ind_ip] - p[ind_im]
								- p[ind_jp] - p[ind_jm]
								- p[ind_kp] - p[ind_km];
					
					alpha += p[index] * s[index];
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_SUM, (*grid).comm);
		alpha = rho[0] / alpha;

		for(int i = 1; i < (*mesh).local_m - 1; i++)
		{
			for(int j = 1; j < (*mesh).local_n - 1; j++)
			{
				for(int k = 1; k < (*mesh).local_q - 1; k++)
				{
					int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;

					x[index] += alpha*p[index];
					r[index] -= alpha*s[index];
					rho[1]   += r[index] * r[index];
				}
			}
		}

		MPI_Allreduce(MPI_IN_PLACE, &rho[1], 1, MPI_DOUBLE, MPI_SUM, (*grid).comm);
		double rho_fraction = rho[1] / rho[0];

		for(int i = 1; i < (*mesh).local_m - 1; i++)
		{
			for(int j = 1; j < (*mesh).local_n - 1; j++)
			{
				for(int k = 1; k < (*mesh).local_q - 1; k++)
				{
					int index = 	i*(*mesh).local_n*(*mesh).local_q + 	j*(*mesh).local_q + k;

					p[index] = r[index] + rho_fraction * p[index];
				}
			}
		}

		rho[0] = rho[1];
		rho[1] = 0;


		//start point exchange
		MPI_Request requests[3*2*2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
										MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
		//exchange row-wise
		if( (*grid).my_row != 0 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[0][0], (*grid).rowm1, 0, (*grid).comm, &requests[0]);
			MPI_Irecv( p, 1, (*mesh).recv_types[0][1], (*grid).rowm1, 0, (*grid).comm, &requests[1]);
		}
		if( (*grid).my_row != (*grid).dim_sizes[0]-1 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[0][1], (*grid).rowp1, 0, (*grid).comm, &requests[2]);
			MPI_Irecv( p, 1, (*mesh).recv_types[0][0], (*grid).rowp1, 0, (*grid).comm, &requests[3]);
		}

		//exchange column-wise
		if( (*grid).my_col != 0 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[1][0], (*grid).colm1, 0, (*grid).comm, &requests[4]);
			MPI_Irecv( p, 1, (*mesh).recv_types[1][1], (*grid).colm1, 0, (*grid).comm, &requests[5]);			
		}
		if( (*grid).my_col != (*grid).dim_sizes[1]-1 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[1][1], (*grid).colp1, 0, (*grid).comm, &requests[6]);
			MPI_Irecv( p, 1, (*mesh).recv_types[1][0], (*grid).colp1, 0, (*grid).comm, &requests[7]);
		}

		//exchange depth-wise
		if( (*grid).my_dep != 0 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[2][0], (*grid).depm1, 0, (*grid).comm, &requests[8]);
			MPI_Irecv( p, 1, (*mesh).recv_types[2][1], (*grid).depm1, 0, (*grid).comm, &requests[9]);			
		}
		if( (*grid).my_dep != (*grid).dim_sizes[2]-1 )
		{
			MPI_Isend( p, 1, (*mesh).send_types[2][1], (*grid).depp1, 0, (*grid).comm, &requests[10]);
			MPI_Irecv( p, 1, (*mesh).recv_types[2][0], (*grid).depp1, 0, (*grid).comm, &requests[11]);
		}


		//print status occasionally and at the end
		static int count = 0;
		count++;
		if((*grid).my_rank == 0 && (count == 1 || count%200 == 0))
			printf("%d: %e\n", count, sqrt(rho[0]));
		else if((*grid).my_rank == 0 && rho[0] <= (*mesh).threshold)
			printf("%d: %e\n", count, sqrt(rho[0]));
		if(count == (*mesh).m*(*mesh).n*(*mesh).q && sqrt(rho[0]) > (*mesh).threshold)
			printf("Reached maximum number of iterations but not threshold.\n");

		MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

		if(count == (*mesh).m*(*mesh).n*(*mesh).q)
			break;

	}while(sqrt(rho[0]) > (*mesh).threshold);

	MPI_Barrier(MPI_COMM_WORLD);
	free(s);
	free(p);

	return;
}


void write_to_file(MESH_T *mesh, GRID_INFO_T *grid)
{//write results to file
	MPI_File fh;

	//if the file exists, overwrite it by deleting it completely and creating it again
	MPI_File_open( (*grid).comm, "results", MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh );
	MPI_File_close( &fh );

	MPI_File_open( (*grid).comm, "results", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh );


	//create and commit an MPI type for the whole 3d mesh for writing to file
	int global_sizes[3] = { (*mesh).m, (*mesh).n, (*mesh).q };
	int local_sizes[3] = { (*mesh).local_m-2, (*mesh).local_n-2, (*mesh).local_q-2 }; //-2: boundaries shall not be included
	int starts[3] = { 0, 0, 0 };
	for(int i = 0; i < (*grid).my_row; i++)
		starts[0] += (*mesh).local_ms[i];
	for(int j = 0; j < (*grid).my_col; j++)
		starts[1] += (*mesh).local_ns[j];
	for(int k = 0; k < (*grid).my_dep; k++)
		starts[2] += (*mesh).local_qs[k];

	MPI_Datatype filewritetype;
	MPI_Type_create_subarray( 3, global_sizes, local_sizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filewritetype );
	MPI_Type_commit(&filewritetype);

	MPI_File_set_view( fh, 0, MPI_DOUBLE, filewritetype, "native", MPI_INFO_NULL );


	//create and commit an MPI type to exclude the (inner and outer) boundary points that are stored in (*mesh).u as well
	global_sizes[0] = (*mesh).local_m;
	global_sizes[1] = (*mesh).local_n;
	global_sizes[2] = (*mesh).local_q;

	starts[0] = 1;
	starts[1] = 1;
	starts[2] = 1;

	MPI_Datatype type_wo_bound;
	MPI_Type_create_subarray( 3, global_sizes, local_sizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type_wo_bound );
	MPI_Type_commit(&type_wo_bound);


	//write to file
	MPI_File_write_all( fh, (*mesh).u, 1, type_wo_bound, MPI_STATUS_IGNORE);


	MPI_File_close( &fh );

	return;
}


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	double start, finish;
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	GRID_INFO_T grid = { .dim_sizes = {0} };
	setup_grid(&grid);

	if(grid.g_rank == 0 && argc != 10)
	{
		fprintf(stderr, "Wrong number of arguments: argc=%d, needed are 10.\n", argc);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MESH_T mesh;
	enum Algorithm alg = NONE;
	read_input(&mesh, &alg, argv);
	mesh.func_f = &f_sink;
	mesh.func_g = &g_source;
	setup_mesh(&mesh, &grid);
	init_mesh(&mesh, &grid);

	switch(alg)
	{
		case JACOBI:
			jacobi(&mesh, &grid);
			break;
		case GS:
			gauss_seidel(&mesh, &grid);
			break;
		case SOR:
			gs_sor(&mesh, &grid);
			break;
		case CG:
			conjugate_gradient(&mesh, &grid);
			break;
		case NONE:
			if(grid.g_rank == 0)
				fprintf(stderr, "No algorithm specified.\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();

	if(grid.g_rank == 0)
		printf("Time needed: %f s\n", finish-start);

	write_to_file(&mesh, &grid);


	MPI_Finalize();
	return 0;
}
