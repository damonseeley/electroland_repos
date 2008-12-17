package net.electroland.connection.animation;

import net.electroland.connection.core.Light;

public class Particle{
	float y, yvec;
	int x, element, gy;
	public boolean done;
	int age, id;
	
	public Particle(int _x, float _y, float _yvec){
		x = _x;
		y = _y;
		yvec = _yvec;
		age = 255;
	}
	public Particle(int _id, int _x, float _y, float _yvec){
		id = _id;
		x = _x;
		y = _y;
		yvec = _yvec;
		age = 255;
	}
	
	public void move(Light[] lights){
		gy = Math.round(y*28);	// position relative to light grid long-ways axis
		if(gy == 28){
			gy = 27;
		} else if(gy < 0){
			gy = 0;
		}
		element = gy*6 + x;
		if(element >= 0 && element < 28*6){
			//lights[element].setValue((int)(Math.random()*age));	// flickering mode
			lights[element].setValue(age);							// smooth fade mode
		}
		
		age -= 5;
		if(age <= 0){
			done = true;
			//spentparticles += 1;
		} else {
			yvec *= 0.9;				
			if(yvec < 0 && y > 0){
				y += yvec;
			} else if(yvec > 0 && y < 1){
				y += yvec;
			} else {
				done = true;
				//spentparticles += 1;
			}
		}
	}
}
