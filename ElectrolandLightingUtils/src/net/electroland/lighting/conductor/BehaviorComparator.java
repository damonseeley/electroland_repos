package net.electroland.lighting.conductor;

import java.util.Comparator;

public class BehaviorComparator implements Comparator<Behavior>{

	public int compare(Behavior a, Behavior b) {
		return a.getPriority() - b.getPriority();
	}
}