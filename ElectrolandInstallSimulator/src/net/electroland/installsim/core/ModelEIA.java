package net.electroland.installsim.core;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
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

	public int spawnPoint[] = {610,25,0}; //no longer need these values, change later

	private int pID = 0;
	private double moveInc = 0.2;
	private double moveRnd = 0.2;

	private double spawnTimeout = 400; //min time between spawn, in ms
	private double spawnTime = System.currentTimeMillis();
	private double spawnChance = 0.0055; //chance of spawning a new person

	private double dScale;

	private IOManager eio;

	public ModelEIA(int populationSize, float sRate, SoundControllerP5 sc) {
		popSize = populationSize;
		spawnRate = sRate; // number of frames between people spawning, on average
		this.sc = sc;

		dScale = 1.9; // hack way of spacing things out
		spawnPoint[0] = (int)(636*dScale);
		spawnPoint[1] = (int)(20*dScale);

		logger.info("Spawnpoint = " + spawnPoint[0]+ "," + spawnPoint[1]);

		initPeople();
		initSensors();
	}


	public void initPeople(){

		// starting first person
		//Person p = new Person(pID,startSpawnPt[0],startSpawnPt[1],startSpawnPt[2], (float)(moveInc+Math.random()*moveRnd));
		Person p = getNewPerson();
		p.setVector(leftVec);
		people.put(pID, p);
		pID++;

		logger.info("PEOPLE: " + people.toString());
	}

	private void initSensors() {
		eio = new IOManager();
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

		for (IOState state : eio.getStates())
		{
			Point3d l = state.getLocation();
			PhotoelectricTripWire s = new PhotoelectricTripWire(state.getID(),(int)(l.x*dScale),(int)(l.y*dScale),0,sensorBox,sc);
			sensors.add(s);
			sensNum++;
		}

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
			if (p.y < 0 || p.y > spawnPoint[1]) {
				people.remove(p.id);
			}
			if (p.x < 0 || p.x > spawnPoint[0]) {
				people.remove(p.id);
			}


		}

		if ((System.currentTimeMillis() - spawnTime) > spawnTimeout){
			if (people.size() < popSize) {
				//check this math
				if (Math.random() < spawnChance) {
					// create a down-walking person
					//Person p = new Person(pID,(int)(startSpawnPt[0]*dScale),(int)(startSpawnPt[1]*dScale),(int)(startSpawnPt[2]*dScale),(float)(moveInc+Math.random()*moveRnd));
					Person p = getNewPerson();
					p.setVector(leftVec);
					people.put(pID, p);
					pID++;
					spawnTime = System.currentTimeMillis();
				}
			}

		} else {
			//nothing
		}

	}

	private Person getNewPerson() {
		return new Person(pID,spawnPoint[0],spawnPoint[1],spawnPoint[2],(float)(moveInc+Math.random()*moveRnd));
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

	int margin;

	public void render(Graphics g) {

		//logger.info("ModelEIA render loop");

		margin = 10;
		BufferedImage bi = new BufferedImage(spawnPoint[0]+margin*2, spawnPoint[1]+margin*2, BufferedImage.TYPE_INT_RGB);


		//clear(g);
		//Graphics2D g2 = (Graphics2D)g;
		Graphics2D g2 = (Graphics2D)bi.getGraphics();
		g2.setColor(Color.WHITE);  
		g2.fillRect(0, 0, bi.getWidth(), bi.getHeight());

		//set styles
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		//g2.setStroke(dashed);

		//render spawn locations
		g2.setColor(new Color(128, 128, 255));
		g2.drawRect(spawnPoint[0]-9,spawnPoint[1],20,2);
		g2.drawRect(spawnPoint[0],spawnPoint[1]-9,2,20);


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

		g.drawImage(bi, margin, margin, null);

	}



}
