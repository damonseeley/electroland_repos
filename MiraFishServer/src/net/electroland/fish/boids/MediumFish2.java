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

public class MediumFish2 extends StandardFish {
//	FishPolyRegion fpr;
	public MediumFish2(Pond pond) {
		super(pond,1f, 2);
		this.species = FishIDs.MEDIUMFISH2_ID;

		this.setSize(80);
		teleport(new Vector3f(
				pond.bounds.getLeft() +  pond.bounds.getWidth() * (float) Math.random(), 
				pond.bounds.getTop() + pond.bounds.getHeight() * (float) Math.random(), 1f));

		setVelocity(new Vector3f(50 - 20*(float)Math.random(),50 - 20*(float)Math.random(),0));


		setSubFlock(this);

		maxSpeed.setSpeed(Util.plusOrMinus(FishProps.THE_FISH_PROPS.getProperty("MediumFish2MaxSpeed", 75), 10));


		//forces
		


		
		add("friction", new Friction(1f, this, .0001f));

		
		
		
		



		add("dartOnTouch", new DartOnTouch(50f, this, 350, 500));

		pond.add(this);


	}

	public static void generate(Pond p, int cnt) {
		for(int i = 0; i < cnt; i++) {
			new MediumFish2(p);
		}
	}


	

}
