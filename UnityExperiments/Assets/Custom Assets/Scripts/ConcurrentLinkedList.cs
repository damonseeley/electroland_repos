using UnityEngine;
using System.Collections;

public class ConcurrentLinkedList<T> {
	// assumes only one thread is putting and one is getting
	private T head;
	private T tail;
	
	public class Element {
		T value;
		Element next = null;

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
					head = head.next;
					return value;
				}
		} else {
			return null;
		}			
	}

}
