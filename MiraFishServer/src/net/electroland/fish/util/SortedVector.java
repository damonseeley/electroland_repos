package net.electroland.fish.util;

import java.util.Vector;

/*
 * will keep vector sorted if you add with a comparable
 */

public class SortedVector<T extends Comparable<T>> extends Vector<T> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public boolean add(T b) {
		for(int i = 0; i < size(); i++) {
			if(b.compareTo(get(i)) <= 0) {
				add(i,b);
				return true;
			}
		}
		add(size(), b);

		return true;
	}


}
