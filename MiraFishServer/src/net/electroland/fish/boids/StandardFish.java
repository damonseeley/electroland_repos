package net.electroland.fish.boids;

import net.electroland.fish.behaviors.AvoidBigger;
import net.electroland.fish.behaviors.AvoidSameSpecies;
import net.electroland.fish.behaviors.MatchFlockVelocity;
import net.electroland.fish.behaviors.MoveToFlockCenter;
import net.electroland.fish.constraints.FaceVelocity;
import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.constraints.NoEntryMap;
import net.electroland.fish.constraints.WorldBounds;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Maps;
import net.electroland.fish.core.Pond;
import net.electroland.fish.forces.AvoidTouch;
import net.electroland.fish.forces.ForceMap;
import net.electroland.fish.forces.MaintainSwimDepth;
import net.electroland.fish.forces.RandomTurn;
import net.electroland.fish.forces.Wave;
import net.electroland.fish.util.FishProps;
import net.electroland.fish.util.Util;

public class StandardFish extends Boid {
	MaxSpeed maxSpeed;
	
	public StandardFish(Pond pond, float scale, int flockId) {
		super(pond, scale, flockId);

		
		this.scaleCompression = FishProps.THE_FISH_PROPS.getProperty("zScaleCompression", .333f); //scale by depth
		
		
		
		// constraints
		add("faceVelocity", new FaceVelocity(10,.1f));
		add("noEntry", new NoEntryMap(5, Maps.NO_ENTRY));
		add("worldBounds", new WorldBounds(pond.bounds));
		add("maxSpeed", maxSpeed= new MaxSpeed(1,20));
		
		
		
		
		//force
		add("wave", new Wave(1000000,this, Util.plusOrMinus(2000, 500),1000));
		
		// DS SITE CHANGE
		//add("avoidBigger", new AvoidBigger(5f, this, Util.plusOrMinus(10, 3f), Util.plusOrMinus(3f,1f)));
		add("avoidBigger", new AvoidBigger(5f, this, Util.plusOrMinus(5, 10f), Util.plusOrMinus(10f,1f)));

		add("avoidCollision", new AvoidSameSpecies(3f, this, Util.plusOrMinus(20f, 3f), Util.plusOrMinus(.30f,.05f)));

		//ds
		//add("rotate", new ForceMap(5, this, Maps.FORCES,.006f)); //orig
		//add("rotate", new ForceMap(55, this, Maps.FORCES,.006f)); //attempt to make edges work better
		add("rotate", new ForceMap(5, this, Maps.FORCES,.006f)); //faster move
		

		if(! (this instanceof BigFish1)) {
			add("avoidTouch", new AvoidTouch(3f, this, 100f));
			add("randomTurn", new RandomTurn(1f, this, 15000, 5000, 1000, 2f));
			if(! (this instanceof JellyFish1)) {
			add("moveToFlockCenter", new MoveToFlockCenter(FishProps.THE_FISH_PROPS.getProperty("StandardMoveToCenterW", 1.2f), this, .02f));
			add("matchFlockVelocity", new MatchFlockVelocity(1f, this));
			}

			add("maintainSwimDepth", new MaintainSwimDepth(2.0f, this, 
					FishProps.THE_FISH_PROPS.getProperty("minSwimDepth", .5f),
					FishProps.THE_FISH_PROPS.getProperty("maxSwimDepth", 1.5f),
					.02f));
		} else {
			//add("avoidTouch", new AvoidTouch(1f, this, 50f));	
			//add("avoidTouch", new AvoidTouch(1f, this, 15f)); // sunday most
			add("avoidTouch", new AvoidTouch(1f, this, 50f));
		}
		
	}

}
