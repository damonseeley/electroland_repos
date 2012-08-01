package net.electroland.fish.boids;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Maps;
import net.electroland.fish.core.Pond;
import net.electroland.fish.forces.DartOnTouch;
import net.electroland.fish.forces.ForceMap;
import net.electroland.fish.forces.Friction;
import net.electroland.fish.forces.KeepMoving;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;

public class MiraFish1 extends StandardFish {
	
	
	float buffersize = 300f;
	float zBuffersize = .1f;


	public MiraFish1(Pond pond) {
		super(pond, 1f, 16); // the last number is the flock/species
		this.species = FishIDs.SMALLFISH1_ID;
		
		this.setSize(16);
		
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));
		
		setVelocity(new Vector3f(250 - 500*(float)Math.random(),250 - 500*(float)Math.random(),0));


		setSubFlock(this);


		// constraints

		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("MiraFish1MaxSpeed", 100), 20));

		//forces

		add("keepMoving", new KeepMoving(.25f, this, 1f));
		


		
		add("friction", new Friction(1f, this, .0001f));

		
		
		
	

//		add("swimCounterClockwise", new SwimCounterClockwise(5f,this,pond.centerIslandBounds,Util.plusOrMinus(.5f, .1f), .01f));
		add("dartOnTouch", new DartOnTouch(50f, this, 100, 500));

		pond.add(this);
	}


	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			MiraFish1 mf = new MiraFish1(p);		
//			System.out.println("creating mira fish " + mf.broadcastFish.objectType);
		}
	}


}
