package net.electroland.installsim.core;
import java.awt.Color;
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
	
	private int lowerSpawnPt[] = {100,600,0};
	private int upperSpawnPt[] = {100,100,0};
	
	private int pID = 0;
	private int moveInc = 1;


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
			p.setLoc(p.x + moveInc*p.getVec()[0], p.y + moveInc*p.getVec()[1]);
		
		}
		
		
		
	}



}
