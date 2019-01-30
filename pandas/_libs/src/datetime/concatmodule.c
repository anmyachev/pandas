#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject*
concat_date_cols(PyObject *self, PyObject *args)
{
    PyObject *sequence = NULL;
    Py_ssize_t sequence_size = 0;

    if (!PyArg_ParseTuple(args, "O", &sequence))
        return NULL;

    if (!PySequence_Check(sequence)) {
        PyErr_SetString(PyExc_RuntimeError, "argument for function concat_date_cols must be sequence");
        return NULL;
    }

    sequence_size = PySequence_Size(sequence);
    if (sequence_size == 1) {
        PyArrayObject *array = (PyArrayObject *) PyArray_ContiguousFromAny(PySequence_GetItem(sequence, 0), NPY_OBJECT, 1, 1);
        if (PyErr_Occurred() != NULL) return NULL;
        Py_ssize_t array_size = PyArray_SIZE(array);
        for (Py_ssize_t i = 0; i < array_size; ++i) {
            PyArray_SETITEM(array, PyArray_GETPTR1(array, i), PyUnicode_FromObject(PyArray_GETITEM(array, (char*)PyArray_GETPTR1(array, i))));
        }
        if (PyErr_Occurred() != NULL) return NULL;
        else return (PyObject*)array;
    } else {
        PyErr_SetString(PyExc_NotImplementedError, "not implemented");
        return NULL;
    }



    /*for (Py_ssize_t i = 0; i < sequence_size; ++i) {
        array = (PyArrayObject *) PyArray_ContiguousFromAny(PySequence_GetItem(sequence, i), NPY_OBJECT, 1, 1);
    }
    if (PyArray_Check(array)) printf("array\n");

    Py_ssize_t array_size = PyArray_SIZE(array);

    for (Py_ssize_t i = 0; i < array_size; ++i) {
         PyArray_SETITEM(array, PyArray_GETPTR1(array, i), PyUnicode_FromObject(PyArray_GETITEM(array, (char*)PyArray_GETPTR1(array, i))));
    }*/

}

static PyMethodDef module_methods[] =
{
     /* name from python, name in C-file, ..., __doc__ string of method */
     {"concat_date_cols", concat_date_cols, METH_VARARGS, "concat date cols and return numpy array"},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "concat",               /* name of module */
    "concat date cols and return numpy array",  /* module documentation, may be NULL */
    -1,                     /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC
PyInit_concat(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}