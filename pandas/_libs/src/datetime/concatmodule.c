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
    }

    if (sequence_size == 1) {
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
        PyObject* fast_array = NULL;
        Py_ssize_t min_array_size = 0;
        for (Py_ssize_t i = 0; i < sequence_size; ++i) {
            array = PySequence_GetItem(sequence, i);
            if (array == NULL) {
                for (Py_ssize_t j = 0; j < i; ++j) Py_DECREF(arrays[j]);
                free(arrays);
                return NULL;
            }
            fast_array = PySequence_Fast(array, "elements of input sequence must be sequence");
            if (fast_array == NULL) {
                Py_DECREF(array);
                for (Py_ssize_t j = 0; j < i; ++j) Py_DECREF(arrays[j]);
                free(arrays);
                return NULL;  //PySequence_Fast set message, which in second argument
            }
            Py_DECREF(array);

            Py_ssize_t array_size = PySequence_Fast_GET_SIZE(fast_array);
            if (min_array_size != 0) {
                min_array_size = (array_size < min_array_size) ? array_size : min_array_size;
            } else {
                min_array_size = array_size;
            }
            arrays[i]  = fast_array;
        }
        npy_intp dims[1];
        dims[0] = min_array_size;
        PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        if (result == NULL) {
            for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
            free(arrays);
            return NULL;
        }

        PyObject* separator = PyUnicode_FromFormat(" ");
        if (separator == NULL) {
            for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
            free(arrays);
            Py_DECREF(result);
            return NULL;
        }
        PyObject* item = NULL;
        PyObject* result_sequence = PyList_New(sequence_size);
        for (Py_ssize_t i = 0; i < min_array_size; ++i) {
            for (Py_ssize_t j = 0; j < sequence_size; ++j) {
                item = PySequence_Fast_GET_ITEM(arrays[j], i);
                if (item == NULL) {
                    Py_DECREF(result_sequence);
                    Py_DECREF(result);
                    for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
                    free(arrays);
                    return NULL;
                }
                Py_INCREF(item);
                PyList_SetItem(result_sequence, j, item);
            }
            PyObject* result_string = PyUnicode_Join(separator, result_sequence);
            if (result_string == NULL) {
                Py_DECREF(result_sequence);
                Py_DECREF(result);
                for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
                free(arrays);
                return NULL;
            }
            if (PyArray_SETITEM(result, PyArray_GETPTR1(result, i), result_string) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "Cannot set resulting item");
                Py_DECREF(result_sequence);
                Py_DECREF(result);
                Py_DECREF(result_string);
                for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
                free(arrays);
                return NULL;
            }
            Py_DECREF(result_string);
        }
        Py_DECREF(result_sequence);
        for (Py_ssize_t i = 0; i < sequence_size; ++i) Py_DECREF(arrays[i]);
        free(arrays);
        return (PyObject*)result;
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
