using System.Collections;
using System.Threading;

public class ConcurrentLinkedList<T> {
	
	/*
	* ConcurrentLinkedList.cs
	* by Eitan Mendelowitz, 6-11-2010
	*
	* Concurrent list for passing tracking data from threaded server to parser.
	* Assumes only one thread is putting and one is getting.
	*/
	
	private Element head;
	private Element tail;
	
	public class Element {
		public T value;
		public Element next = null;

		public Element(T value) {
			this.value = value;			
		}
	}
	
	public void put(T value) {
		if(tail == null) {
			Element newElement = new Element(value);
			lock(newElement) {
				tail = newElement;
				head = newElement;
			}
		} else {
			lock(tail) {
				Element newElement = new Element(value);
				lock(newElement) {
					tail.next = newElement;
					//MonoBehaviour.print(value);
					tail = newElement;
				}				
			}
		}
	}
	
	// will return null if empty
	public T get() {
		if(head != null) {
			lock(head) {
				T value = head.value;
				//MonoBehaviour.print(value);
				if(head.next == null) {// then tail must == head so tail is locked
 					head = null;
					tail = null;
				} else {
					head = head.next;
				}
				return value;
			}
		} else {
			return default(T);
		}			
	}
	
	
}
