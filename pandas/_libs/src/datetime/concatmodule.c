#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject*
concat_date_cols(PyObject *self, PyObject *args)
{
    PyObject *sequence = NULL;
    Py_ssize_t sequence_size = 0;

    if (!PyArg_ParseTuple(args, "O", &sequence)) {
        return NULL;
    }
    if (!PySequence_Check(sequence)) {
        PyErr_SetString(PyExc_TypeError, "argument for function concat_date_cols must be sequence");
        return NULL;
    }

    sequence_size = PySequence_Size(sequence);
    if (sequence_size == -1) {
        return NULL;
    }

    if (sequence_size == 1) {
        Py_ssize_t first_elem = 0;
        PyObject* array = PySequence_GetItem(sequence, first_elem);
        if (array == NULL) {
            return NULL;
        }
        PyObject* fast_array = PySequence_Fast(array, "elements of input sequence must be sequence");
        if (fast_array == NULL) {
            return NULL;  //PySequence_Fast set message, which in second argument
        }

        Py_ssize_t array_size = PySequence_Fast_GET_SIZE(fast_array);
        PyObject* temp = NULL;
        int result = 0;

        for (Py_ssize_t i = 0; i < array_size; ++i) {
            temp = PySequence_Fast_GET_ITEM(fast_array, i);
            if (temp == NULL) {
                return NULL;
            }
            if (!PyUnicode_Check(temp)) {
                temp = PyUnicode_FromObject(temp);
                if (temp == NULL) {
                    return NULL;
                }
                result = PySequence_SetItem(array, i, temp);
                if (result == -1) {
                    PyErr_SetString(PyExc_RuntimeError, "error at unicode item set");
                    return NULL;
                }
                Py_DECREF(temp);
            }
        }
        Py_DECREF(array);
        return PyArray_FROM_O(fast_array);
        /*PyArrayObject *array = (PyArrayObject *) PyArray_ContiguousFromAny(temp, NPY_OBJECT, 1, 1);
        if (PyErr_Occurred() != NULL) {
            return NULL;
        }
        Py_ssize_t array_size = PyArray_SIZE(array);
        for (Py_ssize_t i = 0; i < array_size; ++i) {
            PyArray_SETITEM(array, PyArray_GETPTR1(array, i), PyUnicode_FromObject(PyArray_GETITEM(array, (char*)PyArray_GETPTR1(array, i))));
        }
        if (PyErr_Occurred() != NULL) {
            return NULL;
        }
        else { 
            return (PyObject*)array;
        }*/
    } else {
        PyArrayObject ** arrays;
        Py_ssize_t min_array_size = PyArray_SIZE(*arrays);
        for (Py_ssize_t i = 0; i < sequence_size; ++i) {
            *(arrays + i) = (PyArrayObject *) PyArray_ContiguousFromAny(PySequence_GetItem(sequence, i), NPY_OBJECT, 1, 1);

        }
        if (PyErr_Occurred() != NULL) {
            return NULL;
        }
        return NULL;
        //for (Py_ssize_t i = 1; i < sequence_size)
    }

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