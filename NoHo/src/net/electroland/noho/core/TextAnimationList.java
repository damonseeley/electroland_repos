package net.electroland.noho.core;

import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Vector;


/***
 * handels iteratoring over all ambiens in list AND reaping destroyed ambients (at the same time)
 * @author eitan
 *
 */
public class TextAnimationList {
	LinkedList<TextAnimation> list = new LinkedList<TextAnimation>();
	Vector<TextAnimation> addBuffer = new Vector<TextAnimation>();
	
	public void render() {
		if(! addBuffer.isEmpty()) {
			list.addAll(addBuffer);
			addBuffer.clear();			
		}
		//System.out.println("in ambientlist render");
		if(list.isEmpty()) return;
		ListIterator<TextAnimation> i = list.listIterator(0);
		while(i.hasNext()) {
			TextAnimation ta = i.next();
			if(ta.isDestroyed()) {
				i.remove();
			} else {
				// do we really need render instead of process?
				// ta.render();
				ta.process();
			}
		}

	
	}
	
	public void add(TextAnimation ta) {
		addBuffer.add(ta);
	}
	public int size() {
		return list.size();
	}
	

	public void clear() {
		list.clear();
	}
}

