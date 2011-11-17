package net.electroland.installsim.core;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import javax.vecmath.Point3d;

import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.installsim.sensors.PhotoelectricTripWire;
import net.electroland.installsim.sensors.Sensor;

import org.apache.log4j.Logger;


public class ModelEIA extends ModelGeneric {

	static Logger logger = Logger.getLogger(ModelEIA.class);

	public ConcurrentHashMap<Integer, Person> people = new ConcurrentHashMap<Integer, Person>();
	public Vector<Sensor>sensors = new Vector<Sensor> ();

	public static SoundControllerP5 sc;

	//UI values
	public static float xScale = 1.0f;
	public static float yScale = xScale;
	public static float xOffset = 80.0f;
	public static float yOffset = 70.0f;

	int popSize;
	float spawnRate;

	public static Integer DUMMYID = -1; 

	//private float upVec[] = {0.0f,-1.0f,0.0f};
	//private float dnVec[] = {0.0f,1.0f,0.0f};
	private float leftVec[] = {-1.0f,0.0f,0.0f};

	public int startSpawnPt[] = {600,25,0};

	private int pID = 0;
	private double moveInc = 0.4f;
	private double moveRnd = 1.0f;

	private int spawnTimeout = 30;
	private int spawnTimer = 0;
	private double spawnChance = 0.1;

	private IOManager eio;

	public ModelEIA(int populationSize, float sRate, SoundControllerP5 sc) {
		popSize = populationSize;
		spawnRate = sRate; // number of frames between people spawning, on average
		this.sc = sc;

		initPeople();
		initSensors();
	}


	public void initPeople(){

		// starting first person
		Person p = new Person(pID,startSpawnPt[0],startSpawnPt[1],startSpawnPt[2], (float)(moveInc+Math.random()*moveRnd));
		p.setVector(leftVec);
		people.put(pID, p);
		pID++;

		logger.info("PEOPLE: " + people.toString());
	}

	private void initSensors() {	
		try {
			eio.load("EIA-EIO.properties");
			eio.start();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//vector is not the right term here, this defines a box
		int[] sensorBox = {0,45,0};

		int sensNum = 0;
		/*
		for (IOState state : eio.getStates())
		{
			Point3d l = state.getLocation();
			PhotoelectricTripWire s = new PhotoelectricTripWire(sensNum,(int)l.x,(int)l.y,0,sensorBox,sc);
			sensors.add(s);
			sensNum++;
		}
		*/

		/*
		//for now setup specific 
		float startx = 90;
		float starty = 80;
		float incy = 20;
		float incx = 0;

		for (int i=0; i<27; i++) {
			PhotoelectricTripWire s = new PhotoelectricTripWire(i,startx+incx*i,starty+incy*i,0,vec,sc);
			sensors.add(s);
		}
		 */
		
		
		logger.info("SENSORS: " + sensors.toString());

	}

	public void update() {
		updatePeople();
		detect();
	}

	public void updatePeople(){
		//logger.info("Update People Loop");
		Enumeration<Person> persons = people.elements();
		while(persons.hasMoreElements()) {
			Person p = persons.nextElement();
			p.setLoc(p.x + p.speed*p.getVec()[0], p.y + p.speed*p.getVec()[1]);
			if (p.y < 5 || p.y > 750) {
				people.remove(p.id);
			}

		}

		if (spawnTimer > spawnTimeout){
			if (people.size() < popSize) {
				//check this math
				//if (Math.random() > (1.0-spawnChance)) {
				// create a down-walking person
				Person p = new Person(pID,startSpawnPt[0],startSpawnPt[1],startSpawnPt[2],(float)(moveInc+Math.random()*moveRnd));
				p.setVector(leftVec);
				people.put(pID, p);
				pID++;
				//}
			}
			spawnTimer = 0;
		} else {
			spawnTimer++;
		}

	}

	public void detect() {
		//logger.info("Detect Loop");
		//detect sensor states
		Iterator s = sensors.iterator();
		while (s.hasNext()) {
			((PhotoelectricTripWire) s.next()).detect(people);
		}
	}

	public void createTestPerson(){
		// Add a test person
		Person newPerson = new Person(DUMMYID, -500, -500, 0,(float)(moveInc+Math.random()*moveRnd));
		people.put(DUMMYID, newPerson);
	}

	public void render(Graphics g) {

		//logger.info("ModelEIA render loop");

		//clear(g);
		Graphics2D g2 = (Graphics2D)g;

		//set styles
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		//g2.setStroke(dashed);

		//render spawn locations
		g.setColor(new Color(128, 128, 255));
		g.drawRect(startSpawnPt[0]-9,startSpawnPt[1],20,2);
		g.drawRect(startSpawnPt[0],startSpawnPt[1]-9,2,20);



		//draw sensors
		Enumeration<Sensor> snsr = sensors.elements();
		while(snsr.hasMoreElements()) {
			PhotoelectricTripWire s = (PhotoelectricTripWire)snsr.nextElement();
			//logger.debug("rendering sensor " + s.id);
			s.render(g2);
		}

		//draw people
		Enumeration<Person> persons = people.elements();
		while(persons.hasMoreElements()) {
			Person p = persons.nextElement();
			p.render(g2, p.id);

		}

	}



}
