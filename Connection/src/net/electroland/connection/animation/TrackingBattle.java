package net.electroland.connection.animation;

import java.util.concurrent.ConcurrentHashMap;

import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class TrackingBattle {
	
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, element, row;					// grid position stuff
	public RowEffect[] roweffects;							// effect in each row
	public ConcurrentHashMap<Integer, Chunk> chunks = new ConcurrentHashMap<Integer, Chunk>();	// active chunks
	public Chunk[] chunklist;								// active chunks
	public int chunkcounter;
	public Particle[] particles;
	public Particle[] newparticles;
	private int[] rows, newrows;							// rows occupied by people
	private boolean colliding;
	
	public TrackingBattle(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(28*6)*2 + 3];					// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		roweffects = new RowEffect[28];						// collision effect for each row
		//chunks = new Chunk[0];								// holds active chunks
		chunkcounter = 0;
		particles = new Particle[0];						// holds active particles
		for(int i=0; i<roweffects.length; i++){			// for each row of lights...
			roweffects[i] = new RowEffect(i*gridx);			// make a row effect
		}
	}
	
	public byte[] draw(Person[] peoplelist){
		rows = new int[0];
		colliding = false;
		for(int i=0; i<peoplelist.length; i++){			// for each person...
			int loc[] = peoplelist[i].getIntLoc();			// get person location in grid
			element = loc[1]*gridx + loc[0];				// light # in array
			row = loc[1]*gridx;								// row person is in
			if(element < lights.length && element >= 0){	// if element is in light array...
				if(peoplelist[i].element != element){		// if person has entered a new cell...
					// PLAY A SIMPLE SOUND?
				}
				if(peoplelist[i].newPerson){				// if a new person enters the space...					
					// LAUNCH CHUNK
					chunks.put(new Integer(chunkcounter), new Chunk(lights, chunkcounter, loc[0], peoplelist[i].y, chunks));	// add a new chunk to the chunk list
					chunkcounter++;
				}
				for(int r=0; r<rows.length; r++){			// for each occupied row...
					if(rows[r] == row){						// if sharing an occupied row...
						colliding = true;					// colliding with someone
					}
				}
				newrows = new int[rows.length+1];			// prepare to extend occupied row list
				System.arraycopy(rows, 0, newrows, 0, rows.length);	// copy list
				rows = newrows;								// overwrite occupied row list
				rows[rows.length-1] = row;					// APPEND the new row to list
				
				if(colliding){								// react to row collision
					roweffects[row/gridx].active = true;	// activate row where collision is happening
				} else {
					for(int n=0; n<gridx; n++){			// draw regular occupied row indicator
						lights[row+n].setBlue(253);			// blue 6 light line
					}
					lights[element].setRed(50);				// highlight element person is in
				}
			}
			colliding = false;								// person walked out of area
		}
		
		for(int i=0; i<roweffects.length; i++){			// for each row effect...
			if(roweffects[i].active){						// if active row...
				roweffects[i].draw();						// draw it
			}
		}
		
		/*
		chunklist = new Chunk[chunks.size()];
		chunks.values().toArray(chunklist);		// create list of chunks to iterate through
		for(int i=0; i<chunklist.length; i++){				// for each active chunk effect...
			// compare each chunk against every other chunk to see if they are colliding
			for(int n=i; n<chunklist.length; n++){
				if(chunklist[i].element == chunklist[n].element){	// CHUNKS ARE COLLIDING (so make particles)
					chunks.remove(new Integer(chunklist[i].id));	// remove chunks
					chunks.remove(new Integer(chunklist[n].id));
					newparticles = new Particle[particles.length+2];	// create a longer array
					System.arraycopy(particles, 0, newparticles, 0, particles.length);	// copy old array
					particles = newparticles;							
					if(chunklist[i].yvel < 0){				// determine vector based on direction chunk was traveling
						tempvec = (float)(0.05*Math.random() + 0.01);
					} else {
						tempvec = (float)(-0.05*Math.random() - 0.01);
					}
					// APPEND new particle
					particles[particles.length-2] = new Particle(chunklist[i].gx, chunklist[i].y, tempvec);
					if(chunklist[n].yvel < 0){				// determine vector based on direction chunk was traveling
						tempvec = (float)(0.05*Math.random() + 0.01);
					} else {
						tempvec = (float)(-0.05*Math.random() - 0.01);
					}
					// APPEND new particle
					particles[particles.length-1] = new Particle(chunklist[n].gx, chunklist[n].y, tempvec);
				}
			}
		}
		
		if(particles.length > 0){							// if there are active particles...
			if(particles[0].done){							// if oldest particle is done...
				newparticles = new Particle[particles.length - 1];	// shorten array
				System.arraycopy(particles, 1, newparticles, 0, particles.length-1);
				particles = newparticles;
			}
			
			for(int i=0; i<particles.length; i++){			// draw remaining active particles
				particles[i].move(lights);
			}
		}
		*/
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	
	
	
	
	
	public class RowEffect{
		
		public int row;
		public int[] rowvalues;
		public int[] newvalues;
		public int age;
		public int death;
		public boolean active;
		public boolean playing;
		
		public RowEffect(int _row){
			row = _row;										// left most element value in row
			age = 0;
			death = 60;
			active = false;
			playing = false;
			rowvalues = new int[6];						// light values for row
			newvalues = new int[6];						// new light values for row
			for(int i=0; i<rowvalues.length; i++){			// for each light in row...
				rowvalues[i] = (int)Math.random()*253;		// create random initial light value
			}
		}
		
		public void draw(){
			for(int i=0; i<gridx; i++){					// for each light in row...
				lights[row+i].setValue(rowvalues[i],0);		// set light to stored value
			}
			System.arraycopy(rowvalues, 0, newvalues, 1, rowvalues.length-1);	// move values over by one light
			newvalues[0] = (int)Math.random()*253;			// create new light value
			rowvalues = newvalues;							// copy to original array
			if(age == 0 && !playing){						// if new encounter AND sound not playing
				playing = true;
				// LAUNCH SIMPLE SOUND HERE
			} else	if(age > death){						// controlled life span
				active = false;
				age = 0;
			} else {
				age += 1;
			}
		}
		
	}
	
}
