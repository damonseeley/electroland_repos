package net.electroland.fish.boids;

import java.util.Vector;

import javax.vecmath.Vector3f;

import net.electroland.broadcast.fish.Fish;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Pond;

public class InvisibleSoundFish extends Boid {
	
	public static final Vector<InvisibleSoundFish> ALL_SOUND_FISH  = new Vector<InvisibleSoundFish>();
	
	public static final int PLAY_WAVE_ACCENT = 15;
	public static final int PLAY_AMBIENT_ACCENT = 16;
	public static final int MUTE_AMBIENT_ACCENT = 17;
	

	public static enum State { NONE, PLAY_WAVE, PLAY_AMBIENT, MUTE_AMBIENT};
	public State state = State.NONE;



	public InvisibleSoundFish(Pond pond, Vector3f position) {
		super(pond, 1f, 15);
		this.setVisionRadius(0);
		this.species = FishIDs.INVISIBLE_ID;
		this.position.set(position);
		ALL_SOUND_FISH.add(this);
		this.offScreen = false;
		pond.add(this);
	}
	
	
	public void move() {
		switch(state) {
		case NONE:
			if(broadcastFish.accent == Fish.NOSTATE) {
				broadcastFish.accent = 0;									
			}
			break;
		case PLAY_WAVE:
			broadcastFish.accent = PLAY_WAVE_ACCENT;		
			state = State.NONE;
			break;
		case PLAY_AMBIENT:
			broadcastFish.accent = PLAY_AMBIENT_ACCENT;		
			state = State.NONE;
			break;
		case MUTE_AMBIENT:
			broadcastFish.accent = MUTE_AMBIENT_ACCENT;		
			state = State.NONE;
			break;
		}
		
		super.move();
	}

	public static void generate(Pond p) {
		Vector3f pos = new Vector3f(
				p.bounds.getLeft() + 100,
				p.bounds.getCenterY(),
				-10);
		new InvisibleSoundFish(p, pos);
		pos.x = p.bounds.getCenterX();
		new InvisibleSoundFish(p, pos);
		pos.x = p.bounds.getRight() - 100;
		new InvisibleSoundFish(p, pos);
	}
	
	public static void playWaveSound() {
		for(InvisibleSoundFish isf : ALL_SOUND_FISH) {
			isf.state = State.PLAY_WAVE;
		}
	}
	public static void playAmbientSound() {
		for(InvisibleSoundFish isf : ALL_SOUND_FISH) {
			isf.state = State.PLAY_AMBIENT;
		}
	}

	public static void muteAmbient() {
		for(InvisibleSoundFish isf : ALL_SOUND_FISH) {
			isf.state = State.MUTE_AMBIENT;
		}
	}


}
