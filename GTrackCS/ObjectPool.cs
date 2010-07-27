using System;
using System.Collections.Generic;

// not used in the end

namespace Electroland
{
	// this class is not thread safe!
	public class ObjectPool<T> where T :new()
	{
		List<T> pool = new List<T>();
		
		public ObjectPool()
		{
		}
		
		public T get() {
			int c = pool.Count;
			if(c == 0) {
				return new T();
			} else {
				T retVal=pool[--c];
				pool.RemoveAt(c);
				return retVal;
			}
		}
		
		public void free(T t) {
			pool.Add(t);	
		}
	}
	
	
}

