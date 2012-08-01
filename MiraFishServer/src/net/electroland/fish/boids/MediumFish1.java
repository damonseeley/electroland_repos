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
import net.electroland.fish.forces.DiveOnMedia;
import net.electroland.fish.forces.ForceMap;
import net.electroland.fish.forces.Friction;
import net.electroland.fish.forces.MoveToPointAndPlayVid;
import net.electroland.fish.forces.SwimCounterClockwise;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;


public class MediumFish1 extends StandardFish {

	//	FishPolyRegion fpr;
	public MediumFish1(Pond pond) {
		super(pond,1.1f, 1);
		this.species = FishIDs.MEDIUMFISH1_ID;
		
		setSubFlock(this);
		

		this.setSize(75);
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));

		setVelocity(new Vector3f(250 - 500*(float)Math.random(),250 - 500*(float)Math.random(),0));


		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("MediumFish1MaxSpeed", 75), 15));

		

		//forces
		

		
		add("friction", new Friction(1f, this, .0001f));

		
		
		
		


		add("swimCounterClockwise", new SwimCounterClockwise(5f,this,pond.centerIslandBounds,Util.plusOrMinus(.5f, .1f), .01f));
		add("dartOnTouch", new DartOnTouch(50f, this, 300, 500));

		pond.add(this);

	}

	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			new MediumFish1(p);
		}
	}

	

}
