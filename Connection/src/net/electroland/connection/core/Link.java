package net.electroland.connection.core;

/**
 * This Link class manages a link between two person objects and is responsible for calculating
 * the light levels between the locations of the two people.
 * 
 * THIS IS NO LONGER IN USE
 * 11-23-08 -aaron
 * 
 * @author Aaron Siegel
 *
 */

public class Link {
	public Integer id;
	public Person personA, personB;
	public long birthdate;
	public double xdiff, ydiff, hypo;
	public int gw, gh;									// grid width and height
	public double gx, gy;								// location of light in grid
	public static float maxDist;
	
	public Link(int _id, Person a, Person b, int gridw, int gridh){
		id = new Integer(_id);
		personA = a;
		personB = b;
		gw = gridw;
		gh = gridh;
		maxDist = 10;	// number of lights between people before link breaks
		birthdate = System.currentTimeMillis();
	}
	
	public void connect(Light[] lights){
		measure();
		drawLine(lights);
	}
	
	public void measure(){
		/*
		if(!ConnectionMain.people.containsKey(personA.id) || !ConnectionMain.people.containsKey(personB.id)){
			//System.out.println("link "+id+" destroyed by dissapearance");
			ConnectionMain.links.remove(id);				// destroy this link instance
		}
		xdiff = personA.x*gw - personB.x*gw;				// mapped to lighting grid
		ydiff = personA.y*gh - personB.y*gh;
		hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
		if(Math.abs(hypo) > maxDist){
			//System.out.println("link "+id+" destroyed by length");
			ConnectionMain.links.remove(id);				// destroy this link instance
		}
		*/
	}
	
	public void drawLine(Light[] lights){					// passes a reference to the lights list		
		int element = 0;
		
		// GETTING INCREDIBLY UNPREDICTABLE RESULTS FROM THIS METHOD
		// NEEDS SERIOUS DEBUGGING
		// this wraps somewhere due to a 0 X value
		
		if(Math.abs(xdiff) > Math.abs(ydiff)){				// if greater horizontal space than long ways...
			
			// this is still bugging out for some reason
			for(int i=0; i<Math.abs(xdiff); i++){			// longer value always steps by one
				if(personA.x < personB.x && personA.y < personB.y){			// upper left quadrant
					gy = (int)(personA.y*gh + i);
					gx = (int)(personA.x*gw + Math.abs(ydiff/xdiff)*i);
					//System.out.println("upper left");
				} else if(personA.x > personB.x && personA.y < personB.y){	// upper right quadrant
					gy = (int)(personA.y*gh + i);
					gx = (int)(personA.x*gw - Math.abs(ydiff/xdiff)*i);
					//System.out.println("upper right");
				} else if(personA.x < personB.x && personA.y > personB.y){	// lower left quadrant
					gy = (int)(personA.y*gh - i);
					gx = (int)(personA.x*gw + Math.abs(ydiff/xdiff)*i);
					//System.out.println("lower left");
				} else {													// lower right quadrant
					gy = (int)(personA.y*gh - i);
					gx = (int)(personA.x*gw - Math.abs(ydiff/xdiff)*i);
					//System.out.println("lower right");
				}
				/*
				if(gx == 0){	// kludge solution to prevent wrapping
					gx = 1;
				}
				*/
				element = (int)(gy*gw+gx);
				if(element >= 0 && element < 168){
					lights[element].setBlue(253);			// turns fully blue if along the rounded path
				}
			}
			
		} else {
			
			for(int i=0; i<Math.abs(ydiff); i++){				
				if(personA.x < personB.x && personA.y < personB.y){			// upper left quadrant
					gy = (int)(personA.y*gh + i);
					gx = (int)(personA.x*gw + Math.abs(xdiff/ydiff)*i);
					//System.out.println("upper left");
				} else if(personA.x > personB.x && personA.y < personB.y){	// upper right quadrant
					gy = (int)(personA.y*gh + i);
					gx = (int)(personA.x*gw - Math.abs(xdiff/ydiff)*i);
					//System.out.println("upper right");
				} else if(personA.x < personB.x && personA.y > personB.y){	// lower left quadrant
					gy = (int)(personA.y*gh - i);
					gx = (int)(personA.x*gw + Math.abs(xdiff/ydiff)*i);
					//System.out.println("lower left");
				} else {													// lower right quadrant
					gy = (int)(personA.y*gh - i);
					gx = (int)(personA.x*gw - Math.abs(xdiff/ydiff)*i);
					//System.out.println("lower right");
				}
				/*
				if(gx == 0){	// kludge solution to prevent wrapping
					gx = 1;
				}
				*/
				element = (int)(gy*gw+gx);
				if(element >= 0 && element < 168){
					lights[element].setBlue(253);			// turns fully blue if along the rounded path
				}
			}
			
		}
	}
	
}
