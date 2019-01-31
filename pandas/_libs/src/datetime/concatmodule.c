#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static int convert_and_set_item(PyObject *item, Py_ssize_t index, PyArrayObject *result)
{
	if (item == NULL) {
		return 0;
	}
	if (!PyUnicode_Check(item)) {
		PyObject *unicode_item = PyUnicode_FromObject(item);
		Py_DECREF(item);
		if (unicode_item == NULL) {
			return 0;
		}
		item = unicode_item;
	}
	if (PyArray_SETITEM(result, PyArray_GETPTR1(result, index), item) != 0) {
		PyErr_SetString(PyExc_RuntimeError, "Cannot set resulting item");
		Py_DECREF(item);
		return 0;
	}
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
			for (Py_ssize_t i = 0; i < array_size; ++i) {
				PyObject *item = PyArray_GETITEM(array, PyArray_GETPTR1((PyArrayObject*)array, i));
				if (!convert_and_set_item(item, i, result)) {
					Py_DECREF(result);
					Py_DECREF(array);
					return NULL;
				}
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