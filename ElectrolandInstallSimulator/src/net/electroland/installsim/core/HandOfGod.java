package net.electroland.installsim.core;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.geom.Point2D;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;


public class HandOfGod {
	
	ConcurrentHashMap<Integer, Person> people;
	int popSize;
	float spawnRate;
	
	//move all this out to Main later
	private float upVec[] = {0.0f,-1.0f,0.0f};
	private float dnVec[] = {0.0f,1.0f,0.0f};
	
	public int lowerSpawnPt[] = {125,700,0};
	public int upperSpawnPt[] = {105,25,0};
	
	private int pID = 0;
	private double moveInc = 0.6f;
	private double moveRnd = 0.5f;
	
	private int spawnTimeout = 30;
	private int spawnTimer = 0;
	private double spawnChance = 0.1;


	public HandOfGod(ConcurrentHashMap ppl, int populationSize, float sRate) {
		people = ppl;
		popSize = populationSize;
		spawnRate = sRate; // number of frames between people spawning, on average
		
		initPeople();
	}
	

	public void initPeople(){
		
		// starting lower person
		Person p = new Person(pID,lowerSpawnPt[0],lowerSpawnPt[1],lowerSpawnPt[2]);
		p.setVector(upVec);
		pID++;
		people.put(pID, p);
		
		// starting lower person
		p = new Person(pID,upperSpawnPt[0],upperSpawnPt[1],upperSpawnPt[2]);
		p.setVector(dnVec);
		pID++;
		people.put(pID, p);
		
		
		System.out.println(people.toString());
			
	}
	
	public void updatePeople(){
		Enumeration<Person> persons = InstallSimMainThread.people.elements();
		while(persons.hasMoreElements()) {
			Person p = persons.nextElement();
			p.setLoc(p.x + (float)(Math.random()*moveInc*p.getVec()[0]), p.y + (float)(Math.random()*moveInc*p.getVec()[1]));

		}

		if (spawnTimer > spawnTimeout){
			if (people.size() < popSize) {
				if (Math.random() > (1.0-spawnChance)) {
					// create a down-walking person
					Person p = new Person(pID,lowerSpawnPt[0],lowerSpawnPt[1],lowerSpawnPt[2]);
					p.setVector(upVec);
					pID++;
					people.put(pID, p);
				} else if (Math.random() < spawnChance){
					// create an up-walking person	
					Person p = new Person(pID,upperSpawnPt[0],upperSpawnPt[1],upperSpawnPt[2]);
					p.setVector(dnVec);
					pID++;
					people.put(pID, p);
				}
			}
			spawnTimer = 0;
		} else {
			spawnTimer++;
		}

	}
	
	public void render(Graphics g) {
		
		//render spawn locations
		g.setColor(new Color(128, 128, 255));
		g.drawRect(lowerSpawnPt[0]-9,lowerSpawnPt[1],20,2);
		g.drawRect(lowerSpawnPt[0],lowerSpawnPt[1]-9,2,20);
		g.drawRect(upperSpawnPt[0]-9,upperSpawnPt[1],20,2);
		g.drawRect(upperSpawnPt[0],upperSpawnPt[1]-9,2,20);
		
	}



}
