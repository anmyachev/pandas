#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <locale.h>
#include <string.h>

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
    Py_ssize_t i;
    for (i = 0; i < size; ++i, ++item) Py_DECREF(*item);
    free(arrays);
    return NULL;
}

static PyObject*
concat_date_cols(PyObject *self, PyObject *args)
{
    PyObject *sequence = NULL;
    PyArrayObject *result = NULL;
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
        result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        return (PyObject*)result;
    } else if (sequence_size == 1) {
        PyObject* array = PySequence_GetItem(sequence, 0);
        Py_ssize_t array_size;
        if (array == NULL) {
            return NULL;
        }
        npy_intp dims[1];
        array_size = PySequence_Size(array);
        if (array_size == -1) {
            Py_DECREF(array);
            return NULL;
        }
        dims[0] = array_size;

        result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        if (result == NULL) {
            Py_DECREF(array);
            return NULL;
        }

        if (PyArray_CheckExact(array)) {
            PyArrayObject *ndarray = (PyArrayObject*)array;
            Py_ssize_t i;
            for (i = 0; i < array_size; ++i) {
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
            Py_ssize_t i;
            if (fast_array == NULL) {
                Py_DECREF(result);
                Py_DECREF(array);
                return NULL;  //PySequence_Fast set message, which in second argument
            }

            for (i = 0; i < array_size; ++i) {
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
        PyObject *array = NULL;
        PyObject **parray = NULL;
        PyObject *fast_array = NULL;
        PyObject *separator = NULL;
        PyObject *item = NULL;
        PyObject *list_to_join = NULL;
        Py_ssize_t min_array_size = 0;
        int all_numpy = 1;
        Py_ssize_t i;
        for (i = 0; i < sequence_size; ++i) {
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
            Py_ssize_t i;
            for (i = 0; i < sequence_size; ++i, ++parray) {
                Py_ssize_t array_size = PyArray_SIZE((PyArrayObject*)(*parray));

                if (array_size < 0) {
                    return free_arrays(arrays, sequence_size);
                }

                if (array_size < min_array_size || min_array_size == 0) min_array_size = array_size;
            }
        } else {
            Py_ssize_t i;
            for (i = 0; i < sequence_size; ++i, ++parray) {
                Py_ssize_t array_size;
                fast_array = PySequence_Fast(*parray, "elements of input sequence must be sequence");
                array_size = (fast_array == NULL) ? -1 : PySequence_Fast_GET_SIZE(fast_array);

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
        result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_OBJECT, 0);
        if (result == NULL) {
            return free_arrays(arrays, sequence_size);
        }

        separator = PyUnicode_FromFormat(" ");
        if (separator == NULL) {
            Py_DECREF(result);
            return free_arrays(arrays, sequence_size);
        }
        list_to_join = PyList_New(sequence_size);
        for (i = 0; i < min_array_size; ++i) {
            PyObject *result_string = NULL;
            parray = arrays;
            if (all_numpy) {
                Py_ssize_t j;
                for (j = 0; j < sequence_size; ++j, ++parray) {
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
                Py_ssize_t j;
                for (j = 0; j < sequence_size; ++j, ++parray) {
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
            result_string = PyUnicode_Join(separator, list_to_join);
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

static int inline parse_4digit(const char* s) {
     const char *ch = s;
     int result = 0;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 1000;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 100;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 10;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0');
     return result;
}

static int inline parse_2digit(const char* s) {
    const char *ch = s;
    int result = 0;
    if (*ch < '0' || *ch > '9') return -1;
    result += (int)(*ch - '0') * 10;
    ch++;
    if (*ch < '0' || *ch > '9') return -1;
    result += (int)(*ch - '0');
    return result;
}

static char not_datelike[sizeof(char) * 256];

static PyObject* does_string_look_like_datetime(PyObject* unused, PyObject* arg) {
    PyObject* str = NULL;
    char* buf = NULL;
    Py_ssize_t length = -1;
    int result = 1;

#if PY_MAJOR_VERSION == 2
    if (!PyString_CheckExact(arg)) {
        if (!PyUnicode_CheckExact(arg)) {
            // arg is not a string, so it's certainly not a datetime-looking string
            Py_RETURN_FALSE;
        }
        str = PyObject_Str(arg);
        if (str == NULL) return NULL;
        arg = str;
    }
    if (PyString_AsStringAndSize(arg, &buf, &length) == -1) {
        Py_XDECREF(str);
        return NULL;
    }
#else
    if (!PyUnicode_CheckExact(arg) || !PyUnicode_IS_READY(arg)) {
        PyErr_SetString(PyExc_ValueError, "does_string_look_like_datetime expects a string");
        return NULL;
    }
    buf = PyUnicode_DATA(arg);
    length = PyUnicode_GET_LENGTH(arg);
#endif

    if (length >= 1) {
        char first = *buf;
        if (first == '0') {
            result = 1;
        } else if (length == 1 && not_datelike[Py_CHARMASK(first)]) {
            result = 0;
        } else {
            char* dot_pos = strchr(buf, '.');
            char *to_parse = buf, *end_point = NULL, *to_free = NULL;
            double parsed;
            if (dot_pos != NULL) {
                struct lconv *locale_data = localeconv();
                const char *decimal_point = locale_data->decimal_point;
                if (decimal_point[0] != '.' || decimal_point[1] != 0) {
                    // Python always uses "." as decimal separator, replace with locale-dependent
                    size_t decimal_len = strlen(decimal_point);
                    to_free = to_parse = (char*)(malloc(length + strlen(decimal_point)));
                    if (to_parse == NULL) {
                        Py_XDECREF(str);
                        return PyErr_NoMemory();
                    }
                    memcpy(to_parse, buf, dot_pos - buf);
                    memcpy(&to_parse[dot_pos - buf], decimal_point, decimal_len);
                    memcpy(&to_parse[dot_pos - buf + decimal_len], dot_pos + 1, length - (dot_pos - buf) - 1);
                }
            }

            errno = 0;
            parsed = strtod(to_parse, &end_point);
            if (end_point != to_parse && errno == 0) {
                // need to check if there's anything left
                for (; *end_point != 0 && Py_ISSPACE(*end_point); ++end_point);
                if (*end_point == 0) {
                    // double parsed okay, now check it
                    result = (parsed >= 1000) ? 1 : 0;
                }
            }

            free(to_free);
        }
    }

    Py_XDECREF(str);
    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
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

    memset(not_datelike, 0, sizeof(not_datelike));
    not_datelike['a'] = not_datelike['A'] = 1;
    not_datelike['m'] = not_datelike['M'] = 1;
    not_datelike['p'] = not_datelike['P'] = 1;
    not_datelike['t'] = not_datelike['T'] = 1;

    return PyModule_Create(&moduledef);
}
