package net.electroland.installsim.sensors;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Enumeration;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.installsim.core.Person;
import net.electroland.installsim.core.SoundControllerP5;

import org.apache.log4j.Logger;

public class PhotoelectricTripWire extends Sensor {
	
	static Logger logger = Logger.getLogger(PhotoelectricTripWire.class);

	public int id;
	public float x;
	public float y;
	public float z;
	public int[] sensingVector;
	public SoundControllerP5 sc;
	
	int bodyWidth = 2;
	int wireStroke = 2;
	Color sensorColor;
	Color wireColor;
	
	private boolean tripped = false;
	Timer timer;
	public int tripTime = 150; //ms
	
	private int value; // 0-255
//	private float uptime;  Unused EGM
	
	//constructor
	public PhotoelectricTripWire(int theid, float locx, float locy, float locz, int[] vector, SoundControllerP5 sc) {
		id = theid; 
		x = locx;
		y = locy;
		z = locz;
		sensingVector = vector;
		this.sc = sc;
		
		sensorColor = new Color(160, 160, 160);
		wireColor = new Color(0, 0, 255);
		
		value = 0;
	}
	

	public void render(Graphics g) {

		if (tripped){
			g.setColor(new Color(255,0,0));
			g.fillRect((int)(x-bodyWidth),(int)(y-bodyWidth),bodyWidth*2,bodyWidth*2);
		} else {
			g.setColor(new Color(160, 160, 160));
		}
		
		//render sensor body
		//g.setColor(sensorColor);
		g.drawRect((int)(x-bodyWidth),(int)(y-bodyWidth),bodyWidth*2,bodyWidth*2);
		g.drawString("s" + this.id, (int)x-40, (int)y+2);
		
		//render tripwire
		//g.setColor(sensorColor);
		g.drawLine((int)x, (int)y, (int)x+sensingVector[0], (int)y+sensingVector[1]);
		
	}
	
	public void detect(ConcurrentHashMap people) {

		if (people != null){
			Enumeration<Person> persons = people.elements();
			
			while(persons.hasMoreElements()) {
				Person p = persons.nextElement();
				//logger.info(p);
				if (Math.abs(p.x - x) < (int)x+sensingVector[0]) {
					// wire stroke is not the right variable here, but has to do for now
					if (Math.abs(p.y - y) < wireStroke*2) {
						trip();
					}
				}
			}
		}
		
	}

	public void trip() {
		//this sould really be an event model and not carea bout soundcontroller here
		if (!tripped) {
			sc.iStateUpdate(id);
			//logger.info("tripped " + id);
		}
		tripped = true;
		// set a timer here for untrip
		// update the timer if tripped again before timer expire
		if (timer == null) {
			timer = new Timer();
		    timer.schedule(new untrip(), tripTime);
		} else {
			timer.cancel();
			timer = new Timer();
			timer.schedule(new untrip(), tripTime);
		}
	}
	
	class untrip extends TimerTask {
	    public void run() {
	      //System.out.println("Time's up!" + this);
	      tripped = false;
	      timer.cancel();
	      timer = null;
	    }
	  }
	
	public String getValueAsString() {
		if (tripped){
			return "FD";
		} else {
			return "00";
		}
	}
	
	public String toString() {
		return id + ":(" + x + ", " + y + ") = " + value;
	}


}
