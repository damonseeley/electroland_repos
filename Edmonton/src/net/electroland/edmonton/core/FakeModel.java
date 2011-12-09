package net.electroland.edmonton.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

import net.electroland.edmonton.core.model.OneEventPerPeriodModelWatcher;
import net.electroland.eio.IState;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.eio.model.ModelWatcher;


public class FakeModel {

	Collection<ModelWatcher> watchers = Collections.synchronizedList(new ArrayList<ModelWatcher>());
	Collection<ModelListener> listeners = new ArrayList<ModelListener>();

	public final void addModelWatcher(ModelWatcher watcher, String name, Collection<IState> states)
	{

	}
	public final void addModelListener(ModelListener listener)
	{
		listeners.add(listener);
	}

	public final void poll()
	{

		ModelWatcher test = new OneEventPerPeriodModelWatcher(500);
		Collection someStates = new ArrayList<IState>();
		//IOState i30 = eio.getStateById("i30");
		//logger.info(i30);
		//logger.info("Got IOState " + i30.getID() + " location " + i30.getLocation());
		///this.addModelWatcher(test, "testWatcher", eio.getIStates());

		for (ModelListener listener : listeners)
		{
			int rndScale = 80;
			int rnd = (int)(rndScale*Math.random());
			String whatToDo = "";
			if (rnd < 1){
				whatToDo = "entry1";
			} else if (rnd < 2){
				whatToDo = "egg1";
			} else if (rnd < 3) {
				whatToDo = "exit1";
			}

			if (whatToDo != "") {
				IState is = new IState(whatToDo, 0, 0, 0, null);
				is.setState(true);
				ModelEvent evt = new ModelEvent(is);
				evt.watcherName = "testModelEvent";
				listener.modelChanged(evt);
			}




		}

		}
	}