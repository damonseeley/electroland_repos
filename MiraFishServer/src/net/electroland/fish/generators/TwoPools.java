package net.electroland.fish.generators;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Pond;
import net.electroland.fish.core.PondGenerator;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;

public class TwoPools implements PondGenerator {

	public void generate(FishProps props, Pond pond) throws ClassNotFoundException, NumberFormatException, IllegalArgumentException, SecurityException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
		pond.generator = this;

		String fishList = props.getProperty("fishList", "MicroFish1,20,TinyFish1,20,SmallFish1,20,MediumFish2,6,MediumFish3,6,JellyFish1,4");

		String[] arr = fishList.split(",");

		int f = 0;
		while(f<arr.length) {
			System.out.println("creating");
			Class fishClass= Class.forName("net.electroland.fish.boids." + arr[f++]);
			Method m = fishClass.getMethod("generate", pond.getClass(), Integer.TYPE);
			Boid.resetSubFlockCount(Integer.parseInt(arr[f++]));
			m.invoke(null, pond, Integer.parseInt(arr[f++]));
		}









	}

	public void setStartPosition(Boid b, Bounds bounds) {
		boolean onRight = true;
		
		if(Math.random() > .5) { // on left
			onRight = false;
		} else {
			b.subFlockId +=100; // so doesn't get mixed up with fish on other size of the devide
		}
		
		double third = Math.random();
		if(third <  .667) { // top or bottom
			float x = (float) Math.random() * (bounds.getWidth() - 120);
			if(onRight) {
				x = bounds.getRight() - x;
			}
			if(third < .333) { // top
				b.teleport(new Vector3f(x, - b.size, 1));
				b.setVelocity(new Vector3f(0, 3 + 3f * (float)Math.random(), 1f));					
			} else { // bot
				b.teleport(new Vector3f(x, bounds.getBottom() + b.size, 1));
				b.setVelocity(new Vector3f(0, -3 - 3f * (float)Math.random(), 1f));									
			}
			
		} else {  
			float x = 0-b.size;
			float vx = 3 + 3f * (float)Math.random();
			if(onRight) {
				x = bounds.getRight() + b.size;
				vx = -vx;
			}
			b.teleport(new Vector3f(x, (float) (bounds.getHeight() * Math.random()), 1));
			b.setVelocity(new Vector3f(vx, 0f,0f));
			
			
		}
	}

}
