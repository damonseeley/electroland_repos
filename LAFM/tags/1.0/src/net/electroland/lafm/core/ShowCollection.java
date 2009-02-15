package net.electroland.lafm.core;

import java.util.Vector;
import java.util.Iterator;

public class ShowCollection {

	private Vector <ShowThread>collection;
	private Vector <ShowCollectionListener>listeners;
	private String id;
	
	public ShowCollection(String id){
		this.id = id;
		this.collection = new Vector<ShowThread>();
		this.listeners = new Vector<ShowCollectionListener>();
	}
	
	final public String getId(){
		return this.id;
	}
	
	final protected void addToCollection(ShowThread show){
		show.collection = this;
		collection.add(show);
	}
	
	public void addListener(ShowCollectionListener listener){
		listeners.add(listener);
	}
	
	final protected void complete(ShowThread show){
		collection.remove(show);
		if (collection.size() == 0){
			Iterator<ShowCollectionListener> i = listeners.iterator();
			while (i.hasNext()){
				i.next().notifyCollectionComplete(this);
			}
		}
	}
}