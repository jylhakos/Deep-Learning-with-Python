vector_sum = x.sum()
vector_norm = sum(x**2)
vector_dotprod = w.dot(x)
vector_outerprod = np.outer(w,x)
mat_mult = vector_outerprod@A