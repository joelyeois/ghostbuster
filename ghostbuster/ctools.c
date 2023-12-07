// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>

static PyObject * where(PyObject *self, PyObject *args) {
  PyArrayObject *rangeX, *rangeY, *rangeZ;
  PyArrayObject *xcoords, *ycoords, *zcoords;
  double tmpX, tmpY, tmpZ, tmpl;
  double minX, maxX, minY, maxY, minZ, maxZ;
  npy_intp i,j,k,l;
  int n;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                        &PyArray_Type, &rangeX, &PyArray_Type, &rangeY, &PyArray_Type, &rangeZ, \
                        &PyArray_Type, &xcoords, &PyArray_Type, &ycoords, &PyArray_Type, &zcoords)) return NULL;
  if (NULL == rangeX) return NULL;
  if (NULL == rangeY) return NULL;
  if (NULL == rangeZ) return NULL;
  if (NULL == xcoords) return NULL;
  if (NULL == ycoords) return NULL;
  if (NULL == zcoords) return NULL;
  
  /* Get the dimensions of the ranges */
  npy_intp nx = PyArray_DIM(rangeX,0);
  npy_intp ny = PyArray_DIM(rangeY,0);
  npy_intp nz = PyArray_DIM(rangeZ,0);
  
  /* Get min/max values of the ranges */
  minX = *(double *) PyArray_GETPTR1(rangeX,0);
  maxX = *(double *) PyArray_GETPTR1(rangeX,nx-1);
  minY = *(double *) PyArray_GETPTR1(rangeY,0);
  maxY = *(double *) PyArray_GETPTR1(rangeY,ny-1);
  minZ = *(double *) PyArray_GETPTR1(rangeZ,0);
  maxZ = *(double *) PyArray_GETPTR1(rangeZ,nz-1);

  /* Get the dimension of the coordinates */
  npy_intp nl = PyArray_DIM(xcoords,0);

  /* Allocate an array for the results and initialize to infinity*/
  double *res=(double *)malloc((size_t) (nl*sizeof(double)));
  for (n=0; n<nl; n++) {
    res[n] = INFINITY;
  }

  /* Create index array that can be filled later */
  PyArrayObject * xindex = (PyArrayObject *) PyArray_SimpleNew(1,&nl,NPY_INT);
  PyArrayObject * yindex = (PyArrayObject *) PyArray_SimpleNew(1,&nl,NPY_INT);
  PyArrayObject * zindex = (PyArrayObject *) PyArray_SimpleNew(1,&nl,NPY_INT);
  
  /* Loop through a set of given coordinates and find their indices i,j,k with closest matching on the grid */
  for ( l=0; l<nl; l++) {
    tmpX = *(double *) PyArray_GETPTR1(xcoords,l);
    tmpY = *(double *) PyArray_GETPTR1(ycoords,l);
    tmpZ = *(double *) PyArray_GETPTR1(zcoords,l);

    /* Check for x coordinate */
    if ((tmpX < minX) || (tmpX > maxX)) {
      *(int *) PyArray_GETPTR1(xindex,l) = -1;
    }
    else {
      for (i=0; i<nx; i++) {
        tmpl = fabs(tmpX - *(double *) PyArray_GETPTR1(rangeX,i));
        if (tmpl < res[l]) {
          *(int *) PyArray_GETPTR1(xindex,l) = i;
          res[l] = tmpl;
        }
      }
      res[l] = INFINITY;
    }

    /* Check for y coordinate */
    if ((tmpY < minY) || (tmpY > maxY)) {
      *(int *) PyArray_GETPTR1(yindex,l) = -1;
    }
    else {
      for ( j=0; j<ny; j++) {
        tmpl = fabs(tmpY - *(double *) PyArray_GETPTR1(rangeY,j));
        if (tmpl < res[l]) {
          *(int *) PyArray_GETPTR1(yindex,l) = j;
          res[l] = tmpl;
        }
      }
      res[l] = INFINITY;
    }

    /* Check for z coordinate */
    if (tmpZ < minZ) {
      *(int *) PyArray_GETPTR1(zindex,l) = 0;
    }
    else if (tmpZ >maxZ) {
      *(int *) PyArray_GETPTR1(zindex,l) = -1;
    }
    else {
      for ( k=0; k<nz; k++) {         
        tmpl = fabs(tmpZ - *(double *) PyArray_GETPTR1(rangeZ,k));
        if (tmpl < res[l]) {
          *(int *) PyArray_GETPTR1(zindex,l) = k;
          res[l] = tmpl;
        }
      }
    }
  }
  PyObject * py = PyTuple_New(3);
  PyTuple_SET_ITEM(py, 0, PyArray_Return(xindex));
  PyTuple_SET_ITEM(py, 1, PyArray_Return(yindex));
  PyTuple_SET_ITEM(py, 2, PyArray_Return(zindex));  
  return py;
}


// Assembles given arrays of points and values on a 3D grid
// =========================================================
static PyObject * interp3d(PyObject *self, PyObject *args) {
  //PyArrayObject *rangeX, *rangeY, *rangeZ;
  PyArrayObject *xcoords, *ycoords, *zcoords;
  PyArrayObject *values;
  double tx,ty,tz,fx,fy,fz, cx,cy,cz;
  double f,v;
  npy_intp l;
  int size;
  long x,y,z;

  if (!PyArg_ParseTuple(args, "O!O!O!O!i",
                        &PyArray_Type, &xcoords, &PyArray_Type, &ycoords, &PyArray_Type, &zcoords, \
                        &PyArray_Type, &values, &size )) return NULL;

  /* Check arguments */
  if (NULL == xcoords) return NULL;
  if (NULL == ycoords) return NULL;
  if (NULL == zcoords) return NULL;
  if (NULL == values)  return NULL;
  
  /* Define shape and center */
  npy_intp shape[3] = {size,size,size};
  long center = size/2;
  
  /* Get the number of given values */
  npy_intp nl = PyArray_DIM(values,0);

  /* Create volume and weights array that can be filled later */
  PyArrayObject * volume  = (PyArrayObject *) PyArray_ZEROS(3, shape, NPY_FLOAT64, 0);
  PyArrayObject * weights = (PyArrayObject *) PyArray_ZEROS(3, shape, NPY_FLOAT64, 0);
  
  /* Loop through a set of given coordinates and find their indices i,j,k with closest matching on the grid */
  for (l=0; l<nl; l++) {

    /* Get coordinates and values */
    tx = *(double *) PyArray_GETPTR1(xcoords,l) + center;
    ty = *(double *) PyArray_GETPTR1(ycoords,l) + center;
    tz = *(double *) PyArray_GETPTR1(zcoords,l) + center;
    v = *(double *) PyArray_GETPTR1(values,l);

    /* Calculate vectors for interpolation */
    x = tx; // rounding down
    y = ty; // rounding down
    z = tz; // rounding down

    /* Checking for outliers */
    if (x < 0 || x > size-2 || y < 0 || y > size-2 || z < 0 || z > size-2) {
      continue;
    }

    fx = tx - x;
    fy = ty - y;
    fz = tz - z;
    cx = 1. - fx;
    cy = 1. - fy;
    cz = 1. - fz;

    f = cx*cy*cz; 
    *(double *) PyArray_GETPTR3(volume,x,y,z) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x,y,z) += f;

    f = cx*cy*fz; 
    *(double *) PyArray_GETPTR3(volume,x,y,z+1) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x,y,z+1) += f;
    
    f = cx*fy*cz; 
    *(double *) PyArray_GETPTR3(volume,x,y+1,z) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x,y+1,z) += f;
    
    f = cx*fy*fz; 
    *(double *) PyArray_GETPTR3(volume,x,y+1,z+1) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x,y+1,z+1) += f;
    
    f = fx*cy*cz; 
    *(double *) PyArray_GETPTR3(volume, x+1,y,z) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x+1,y,z) += f;

    f = fx*cy*fz; 
    *(double *) PyArray_GETPTR3(volume,x+1,y,z+1) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x+1,y,z+1) += f;

    f = fx*fy*cz; 
    *(double *) PyArray_GETPTR3(volume,x+1,y+1,z) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x+1,y+1,z) += f;
    
    f = fx*fy*fz;
    *(double *) PyArray_GETPTR3(volume,x+1,y+1,z+1) += (f * v);
    *(double *) PyArray_GETPTR3(weights,x+1,y+1,z+1) += f;
  }
  
  PyObject * py = PyTuple_New(2);
  PyTuple_SET_ITEM(py, 0, PyArray_Return(volume));
  PyTuple_SET_ITEM(py, 1, PyArray_Return(weights));
  //PyTuple_SET_ITEM(py, 2, PyArray_Return(counts));
  return py;
}

// Slices a given 3D volume into 2D arrays of points and values
// ============================================================
static PyObject * slice3d(PyObject *self, PyObject *args) {
  PyArrayObject *xcoords, *ycoords, *zcoords;
  PyArrayObject *volume;
  double tx,ty,tz,fx,fy,fz, cx,cy,cz;
  double f,v;
  npy_intp l;
  int size;
  long x,y,z;

  if (!PyArg_ParseTuple(args, "O!O!O!O!i",
                        &PyArray_Type, &xcoords, &PyArray_Type, &ycoords, &PyArray_Type, &zcoords, \
                        &PyArray_Type, &volume, &size )) return NULL;

  /* Check arguments */
  if (NULL == xcoords) return NULL;
  if (NULL == ycoords) return NULL;
  if (NULL == zcoords) return NULL;
  if (NULL == volume)  return NULL;
  
  /* Define center */
  long center = size/2;
  
  /* Get the number of given coordinates */
  npy_intp nl = PyArray_DIM(xcoords,0);
  npy_intp shape[1] = {nl};

  /* Create slices array that can be filled later */
  PyArrayObject * slices   = (PyArrayObject *) PyArray_ZEROS(1, shape, NPY_FLOAT64, 0);
  PyArrayObject * weights  = (PyArrayObject *) PyArray_ZEROS(1, shape, NPY_FLOAT64, 0);
  
  /* Loop through a set of given coordinates and find their indices i,j,k with closest matching on the grid */
  for (l=0; l<nl; l++) {

    /* Get coordinates and values */
    tx = *(double *) PyArray_GETPTR1(xcoords,l) + center;
    ty = *(double *) PyArray_GETPTR1(ycoords,l) + center;
    tz = *(double *) PyArray_GETPTR1(zcoords,l) + center;

    /* Calculate vectors for interpolation */
    x = tx; // rounding down
    y = ty; // rounding down
    z = tz; // rounding down

    /* Checking for outliers */
    if (x < 0 || x > size-2 || y < 0 || y > size-2 || z < 0 || z > size-2) {
      continue;
    }

    fx = tx - x;
    fy = ty - y;
    fz = tz - z;
    cx = 1. - fx;
    cy = 1. - fy;
    cz = 1. - fz;

    f = cx*cy*cz; 
    v  = *(double *) PyArray_GETPTR3(volume,x,y,z);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;

    f = cx*cy*fz; 
    v  = *(double *) PyArray_GETPTR3(volume,x,y,z+1);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;
    
    f = cx*fy*cz; 
    v  = *(double *) PyArray_GETPTR3(volume,x,y+1,z);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;
    
    f = cx*fy*fz; 
    v  = *(double *) PyArray_GETPTR3(volume,x,y+1,z+1);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;
    
    f = fx*cy*cz; 
    v  = *(double *) PyArray_GETPTR3(volume,x+1,y,z);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;

    f = fx*cy*fz; 
    v  = *(double *) PyArray_GETPTR3(volume,x+1,y,z+1);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;

    f = fx*fy*cz; 
    v  = *(double *) PyArray_GETPTR3(volume,x+1,y+1,z);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;
    
    f = fx*fy*fz;
    v  = *(double *) PyArray_GETPTR3(volume,x+1,y+1,z+1);
    *(double *) PyArray_GETPTR1(slices,l) += (f * v);
    *(double *) PyArray_GETPTR1(weights,l) += f;
  }
  
  PyObject * py = PyTuple_New(2);
  PyTuple_SET_ITEM(py, 0, PyArray_Return(slices));
  PyTuple_SET_ITEM(py, 1, PyArray_Return(weights));
  return py;
}

static PyMethodDef _CtoolsMethods[] = {
  {"where",    where,    METH_VARARGS, "Find the closest indices of given coordinates on a 3D grid."},
  {"interp3d", interp3d, METH_VARARGS, "Assemble given array of values and coordinates on a 3D grid using interpolation."},
  {"slice3d",  slice3d,  METH_VARARGS, "Slices a given 3D volume based on given coordinates using interpolation."},
  {NULL, NULL, 0, NULL}
};

// Macro to make module definition compatible with python 2 and 3
// taken from: http://python3porting.com/cextensions.html#module-initialization
#if PY_MAJOR_VERSION >= 3
#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)                                                                              
#define MOD_DEF(ob, name, doc, methods) \
  static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; ob = PyModule_Create(&moduledef);
#else
#define MOD_ERROR_VAL
#define MOD_SUCCESS_VAL(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_DEF(ob, name, doc, methods) ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(ctools)
{
  import_array();
  PyObject *m;
  MOD_DEF(m, "ctools", "EMLAB Tools implemented in C", _CtoolsMethods);
  if (m == NULL)
    return MOD_ERROR_VAL;
  return MOD_SUCCESS_VAL(m);
}
