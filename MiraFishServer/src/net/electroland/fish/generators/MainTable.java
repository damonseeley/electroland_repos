package net.electroland.fish.generators;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.vecmath.Vector3f;

import net.electroland.fish.boids.BigFish1;
import net.electroland.fish.boids.InvisibleSoundFish;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Pond;
import net.electroland.fish.core.PondGenerator;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.ContentList;
import net.electroland.fish.util.FishProps;

public class MainTable implements PondGenerator {

	public void generate(FishProps props, Pond pond) throws ClassNotFoundException, NumberFormatException, IllegalArgumentException, SecurityException, IllegalAccessException, InvocationTargetException, NoSuchMethodException, IOException {
		pond.generator = this;
		ContentList.getTheContactList();
		
		InvisibleSoundFish.generate(pond);		
		

		String fishList = props.getProperty("fishList", "BigFish1,30,5,MicroFish1,10,20,TinyFish1,10,20,SmallFish1,10,20,MediumFish2,2,6,MediumFish3,2,6,JellyFish1,1,4");
		
		if(! fishList.equals("")) {
		String[] arr = fishList.split(",");

		
		int f = 0;
		while(f<arr.length) {
			String name =  arr[f++];
			String flockSize = arr[f++];
			String number = arr[f++];
			System.out.println("creating " + number + "  " + name + "  max flock size " + flockSize);
			Class fishClass= Class.forName("net.electroland.fish.boids." + name);
			Method m = fishClass.getMethod("generate", pond.getClass(), Integer.TYPE);
			Boid.resetSubFlockCount(Integer.parseInt(flockSize));
			m.invoke(null, pond, Integer.parseInt(number));
		}
		}

		


		for(int i = 0; i < BigFish1.THEBIGFISH.size(); i++) {
			BigFish1.THEBIGFISH.get(i).setIndex(i+1);

		}





	}

	public void setStartPosition(Boid b, Bounds bounds) {
		if(b.isBigFish) {
			((BigFish1) b).reset();
		}
		double d = Math.random();
		if(d < .25) { // start on top
			b.teleport(new Vector3f(
					(float) Math.random() * bounds.getWidth(),
					-b.size, 1f
			));
			b.setVelocity(new Vector3f(0, 3 + 3f * (float)Math.random(), 0f));					
		} else if(d < .5) { // bottom
			b.teleport(new Vector3f(
					(float) Math.random() * bounds.getWidth(),
					bounds.getBottom() + b.size, 1f
			));
			b.setVelocity(new Vector3f(0, -3 - 3f * (float)Math.random(), 0f));										
		} else if(d < .75) {
			b.teleport(new Vector3f(
					- b.size,
					(float) Math.random() * bounds.getHeight(),
					1f
			));
			b.setVelocity(new Vector3f(3 + 3f * (float) Math.random(), 0f, 0f));
		} else {
			b.teleport(new Vector3f(
					bounds.getRight() + b.size,
					(float) Math.random() * bounds.getHeight(),
					1f
			));
			b.setVelocity(new Vector3f(-3 - 3f * (float) Math.random(), 0f, 0f));

		}		
	}

}
