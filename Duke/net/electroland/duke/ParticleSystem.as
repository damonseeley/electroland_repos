package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.Event;
	//import de.polygonal.ds.HashMap;
	//import de.polygonal.ds.Iterator;
	
	public class ParticleSystem extends MovieClip{
		
		private var people:Array;
		private var particles:Array;
		//private var particles:HashMap;
		private var personID:Number;
		private var particleID:Number;
		private var particleLayer:MovieClip;
		private var personLayer:MovieClip;
		
		/*
		PARTICLESYSTEM.as
		by Aaron Siegel, 2-1-2010
		
		Controls updating the state of the system as a whole, including adding or
		removing Person objects or Particle objects.
		*/
		
		public function ParticleSystem(){
			particleLayer = new MovieClip();
			personLayer = new MovieClip();
			addChild(particleLayer);
			addChild(personLayer);
		}
		
		public function setup(particleCount:Number, personCount:Number):void{
			personID = 0;
			particleID = 0;
			people = new Array();
			particles = new Array();
			//particles = new HashMap();
			for(var i:Number = 0; i<personCount; i++){
				var xPos:Number = Math.random()*stage.stageWidth;
				var yPos:Number = Math.random()*stage.stageHeight;
				var radius:Number = 25;
				var mass:Number = 1;
				var person:Person = new Person(personID, xPos, yPos, radius, mass);
				personLayer.addChild(person);
				people.push(person);
				for(var p:Number = 0; p<particleCount; p++){
					createNewParticle(personID, xPos, yPos);
				}
				personID++;
			}
			
			this.addEventListener(Event.ENTER_FRAME, update);
		}
		
		public function update(e:Event):void{
			/*
			var iter:Iterator = particles.getIterator();
			var particle:Particle;
			while(iter.hasNext()){
				particle = iter.next();
				particle.move();
			}
			iter = particles.getIterator();
			while(iter.hasNext()){
				particle = iter.next();
				particle.applyGravity(people);
			}
			*/
			
			for(var i:Number = 0; i<particles.length; i++){
				particles[i].move();
			}
			for(i=0; i<particles.length; i++){
				particles[i].applyGravity(people);
				//particles[i].applyGravity(particles);
			}
			
		}
		
		public function createNewParticle(emitterID:Number, xPos:Number, yPos:Number):void{
			// emit particles from this point with an initial random vector
			var particle:Particle = new Particle(particleID, emitterID, xPos, yPos, 5, 1);
			particle.addCallBack(removeParticle);
			particleLayer.addChild(particle);
			particles.push(particle);
			//particles.insert(particleID, particle);
			particleID++;
		}
		
		public function removeParticle(e:ParticleEvent){
			// remove particle
			//particles.splice(e.id, 1);
			trace("particle "+ e.id +" removed, "+ particles.length + " left");
		}
		
	}
	
}