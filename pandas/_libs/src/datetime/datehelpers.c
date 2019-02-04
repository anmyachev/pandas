#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static inline int convert_and_set_item(PyObject *item, Py_ssize_t index, PyArrayObject *result)
{
    int needs_decref = 0;
    if (item == NULL) {
        return 0;
    }
#if PY_MAJOR_VERSION == 2
    if (!PyString_Check(item) && !PyUnicode_Check(item)) {
        PyObject *str_item = PyObject_Str(item);
#else
    if (!PyUnicode_Check(item)) {
        PyObject *str_item = PyUnicode_FromObject(item);
#endif
        if (str_item == NULL) {
            return 0;
        }
        item = str_item;
        needs_decref = 1;
    }
    if (PyArray_SETITEM(result, PyArray_GETPTR1(result, index), item) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot set resulting item");
        if (needs_decref) Py_DECREF(item);
        return 0;
    }
    if (needs_decref) Py_DECREF(item);
    return 1;
}

static PyObject* free_arrays(PyObject** arrays, Py_ssize_t size) {
    PyObject** item = arrays;
    for (Py_ssize_t i = 0; i < size; ++i, ++item) Py_DECREF(*item);
    free(arrays);
    return NULL;
}

static PyObject*
concat_date_cols(PyObject *self, PyObject *args)
{
    PyObject *sequence = NULL;
    Py_ssize_t sequence_size = 0;

    if (!PyArg_ParseTuple(args, "O", &sequence)) {
        return NULL;
    }
    if (!PySequence_Check(sequence)) {
        PyErr_SetString(PyExc_TypeError, "argument must be sequence");
        return NULL;
    }

    sequence_size = PySequence_Size(sequence);
    if (sequence_size == -1) {
        return NULL;
    } else if (sequence_size == 0) {
        npy_intp dims[1];
        dims[0] = 0;
        PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        return (PyObject*)result;
    } else if (sequence_size == 1) {
        PyObject* array = PySequence_GetItem(sequence, 0);
        if (array == NULL) {
            return NULL;
        }
        npy_intp dims[1];
        Py_ssize_t array_size = PySequence_Size(array);
        if (array_size == -1) {
            Py_DECREF(array);
            return NULL;
        }
        dims[0] = array_size;

        PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        if (result == NULL) {
            Py_DECREF(array);
            return NULL;
        }

        if (PyArray_CheckExact(array)) {
            PyArrayObject *ndarray = (PyArrayObject*)array;
            for (Py_ssize_t i = 0; i < array_size; ++i) {
                PyObject *item = PyArray_GETITEM(ndarray, PyArray_GETPTR1(ndarray, i));
                if (!convert_and_set_item(item, i, result)) {
                    Py_DECREF(result);
                    Py_DECREF(array);
                    Py_DECREF(item);
                    return NULL;
                }
                Py_DECREF(item);
            }
        } else {
            PyObject* fast_array = PySequence_Fast(array, "elements of input sequence must be sequence");
            if (fast_array == NULL) {
                Py_DECREF(result);
                Py_DECREF(array);
                return NULL;  //PySequence_Fast set message, which in second argument
            }

            for (Py_ssize_t i = 0; i < array_size; ++i) {
                PyObject* item = PySequence_Fast_GET_ITEM(fast_array, i);
                if (!convert_and_set_item(item, i, result)) {
                    Py_DECREF(result);
                    Py_DECREF(array);
                    Py_DECREF(fast_array);
                    return NULL;
                }
            }
            Py_DECREF(fast_array);
        }
        Py_DECREF(array);
        return (PyObject*)result;
    } else {
        PyObject **arrays = (PyObject**) malloc(sizeof(PyObject*) * sequence_size);
        PyObject* array = NULL;
        PyObject** parray = NULL;
        PyObject* fast_array = NULL;
        Py_ssize_t min_array_size = 0;
        int all_numpy = 1;
        for (Py_ssize_t i = 0; i < sequence_size; ++i) {
            array = PySequence_GetItem(sequence, i);
            if (array == NULL) {
                return free_arrays(arrays, i);
            }
            if (PyArray_CheckExact(array)) {
                if (PyArray_NDIM((PyArrayObject*)array) != 1) {
                    PyErr_SetString(PyExc_ValueError, "ndarrays must be 1-dimentional");
                    return free_arrays(arrays, i);
                }
            } else {
                all_numpy = 0;
            }
            arrays[i] = array;
        }

        parray = arrays;
        if (all_numpy) {
            for (Py_ssize_t i = 0; i < sequence_size; ++i, ++parray) {
                Py_ssize_t array_size = PyArray_SIZE((PyArrayObject*)(*parray));

                if (array_size < 0) {
                    return free_arrays(arrays, sequence_size);
                }

                if (array_size < min_array_size || min_array_size == 0) min_array_size = array_size;
            }
        } else {
            for (Py_ssize_t i = 0; i < sequence_size; ++i, ++parray) {
                fast_array = PySequence_Fast(*parray, "elements of input sequence must be sequence");
                Py_ssize_t array_size = (fast_array == NULL) ? -1 : PySequence_Fast_GET_SIZE(fast_array);

                if (array_size < 0) {
                    Py_XDECREF(fast_array);
                    return free_arrays(arrays, sequence_size);
                }
                Py_DECREF(array);
                arrays[i] = fast_array;

                if (array_size < min_array_size || min_array_size == 0) min_array_size = array_size;
            }
        }
        npy_intp dims[1];
        dims[0] = min_array_size;
        PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        if (result == NULL) {
            return free_arrays(arrays, sequence_size);
        }

        PyObject* separator = PyUnicode_FromFormat(" ");
        if (separator == NULL) {
            Py_DECREF(result);
            return free_arrays(arrays, sequence_size);
        }
        PyObject* item = NULL;
        PyObject* list_to_join = PyList_New(sequence_size);
        for (Py_ssize_t i = 0; i < min_array_size; ++i) {
            parray = arrays;
            if (all_numpy) {
                for (Py_ssize_t j = 0; j < sequence_size; ++j, ++parray) {
                    PyArrayObject* arr = (PyArrayObject*)(*parray);
                    item = PyArray_GETITEM(arr, PyArray_GETPTR1(arr, i));
                    if (item == NULL) {
                        Py_DECREF(list_to_join);
                        Py_DECREF(result);
                        return free_arrays(arrays, sequence_size);
                    }
                    PyList_SetItem(list_to_join, j, item);
                }
            } else {
                for (Py_ssize_t j = 0; j < sequence_size; ++j, ++parray) {
                    item = PySequence_Fast_GET_ITEM(*parray, i);
                    if (item == NULL) {
                        Py_DECREF(list_to_join);
                        Py_DECREF(result);
                        return free_arrays(arrays, sequence_size);
                    }
                    Py_INCREF(item);
                    PyList_SetItem(list_to_join, j, item);
                }
            }
            PyObject* result_string = PyUnicode_Join(separator, list_to_join);
            if (result_string == NULL) {
                Py_DECREF(list_to_join);
                Py_DECREF(result);
                return free_arrays(arrays, sequence_size);
            }
            if (PyArray_SETITEM(result, PyArray_GETPTR1(result, i), result_string) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "Cannot set resulting item");
                Py_DECREF(list_to_join);
                Py_DECREF(result);
                Py_DECREF(result_string);
                return free_arrays(arrays, sequence_size);
            }
            Py_DECREF(result_string);
        }
        Py_DECREF(list_to_join);
        (void)free_arrays(arrays, sequence_size);
        return (PyObject*)result;
    }

}

// cdef set _not_datelike_strings = {'a', 'A', 'm', 'M', 'p', 'P', 't', 'T'}

static PyObject* does_string_look_like_datetime(PyObject* unused, PyObject* arg) {
    /* 

cpdef bint _does_string_look_like_datetime(object date_string):
    if date_string.startswith('0'):
        # Strings starting with 0 are more consistent with a
        # date-like string than a number
        return True

    try:
        if float(date_string) < 1000:
            return False
    except ValueError:
        pass

    if date_string in _not_datelike_strings:
        return False

    return True
    */
    return NULL;
}

static PyMethodDef module_methods[] =
{
     /* name from python, name in C-file, ..., __doc__ string of method */
     {"concat_date_cols", concat_date_cols, METH_VARARGS, "concatenates date cols and returns numpy array"},
     {"does_string_look_like_datetime", does_string_look_like_datetime, METH_O, "checks if string looks like a datetime"},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "datehelpers",               /* name of module */
    "helpers for datetime structures manipulation",  /* module documentation, may be NULL */
    -1,                     /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC
PyInit_datehelpers(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
