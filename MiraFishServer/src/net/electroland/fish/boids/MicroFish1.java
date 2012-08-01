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

public class MicroFish1 extends StandardFish {
	float buffersize = 300f;
	float zBuffersize = .1f;


	public MicroFish1(Pond pond) {
		super(pond, 1f, 7);// the last number is the flock/species
		this.species = FishIDs.MICROFISH1_ID;

		this.setSize(5);
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));

		setVelocity(new Vector3f(250 - 500*(float)Math.random(),250 - 500*(float)Math.random(),0));


		setSubFlock(this);

		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("MicroFish1MaxSpeed", 90), 20));

		

		//forces
		

		
		add("friction", new Friction(1f, this, .0001f));

		
		
		
		
		//behaviors
		add("moveToFlockCenter", new MoveToFlockCenter(FishProps.THE_FISH_PROPS.getProperty("StandardMoveToCenterW", 1.5f), this, .01f));
		add("matchFlockVelocity", new MatchFlockVelocity(1f, this));



//		add("swimCounterClockwise", new SwimCounterClockwise(5f,this,pond.centerIslandBounds,Util.plusOrMinus(.5f, .1f), .01f));
		add("dartOnTouch", new DartOnTouch(50f, this, 450, 1000));

		pond.add(this);
	}

	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			generate(p);
		}
	}


	public static Boid generate(Pond p) {
		return new MicroFish1(p);
	}


}
