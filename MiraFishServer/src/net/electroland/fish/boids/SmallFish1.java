package net.electroland.fish.boids;

import javax.vecmath.Vector3f;

import net.electroland.fish.behaviors.AvoidSameSpecies;
import net.electroland.fish.behaviors.MatchFlockVelocity;
import net.electroland.fish.behaviors.MoveToFlockCenter;
import net.electroland.fish.constraints.FaceVelocity;
import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.constraints.NoEntryMap;
import net.electroland.fish.constraints.NoEntryRegion;
import net.electroland.fish.constraints.WorldBounds;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Maps;
import net.electroland.fish.core.Pond;
import net.electroland.fish.forces.AvoidCenter;
import net.electroland.fish.forces.AvoidEdges;
import net.electroland.fish.forces.DartOnTouch;
import net.electroland.fish.forces.ForceMap;
import net.electroland.fish.forces.Friction;
import net.electroland.fish.forces.SwimCounterClockwise;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;

public class SmallFish1 extends StandardFish {
	
	
	float buffersize = 300f;
	float zBuffersize = .1f;


	public SmallFish1(Pond pond) {
		super(pond, 1f, 4); // the last number is the flock/species
		this.species = FishIDs.SMALLFISH1_ID;
		
		this.setSize(32);
		
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));
		
		setVelocity(new Vector3f(250 - 500*(float)Math.random(),250 - 500*(float)Math.random(),0));



		setSubFlock(this);

		// constraints

		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("SmallFish11MaxSpeed", 100), 20));


		//forces
		

		
		add("friction", new Friction(1f, this, .0001f));

		
		
		
		


//		add("swimCounterClockwise", new SwimCounterClockwise(5f,this,pond.centerIslandBounds,Util.plusOrMinus(.5f, .1f), .01f));
		add("dartOnTouch", new DartOnTouch(50f, this, 350, 1000));

		pond.add(this);
	}


	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			new SmallFish1(p);
		}
	}


}
