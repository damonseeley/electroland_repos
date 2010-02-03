package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.Event;
	import com.ericfeminella.collections.HashMap;
	
	public class ParticleSystem extends MovieClip{
		
		private var people:Array;
		private var particles:HashMap;
		private var personID:Number;
		private var particleID:Number;
		private var particleLayer:MovieClip;
		private var personLayer:MovieClip;
		private var colors:Array;	// preset colors
		
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
			
			colors = new Array();
			var color1:Array = new Array();	// cyan
			color1.push(0);
			color1.push(150);
			color1.push(255);
			color1.push(0.8);
			var color2:Array = new Array();	// orange
			color2.push(255);
			color2.push(100);
			color2.push(0);
			color2.push(0.8);
			var color3:Array = new Array();	// green
			color3.push(0);
			color3.push(200);
			color3.push(0);
			color3.push(0.8);
			
			colors.push(color1);	// preset colors
			colors.push(color2);
			colors.push(color3);
		}
		
		public function setup(particleCount:Number, personCount:Number):void{
			personID = 0;
			particleID = 0;
			people = new Array();
			particles = new HashMap();
			for(var i:Number = 0; i<personCount; i++){
				var xPos:Number = Math.random()*stage.stageWidth;
				var yPos:Number = Math.random()*stage.stageHeight;
				var radius:Number = 25;
				var mass:Number = 1;
				var person:Person = new Person(personID, xPos, yPos, radius, mass);
				person.setParticleColor(colors[i]);
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
			
			var values:Array = particles.getValues();
			for(var i:Number = 0; i<values.length; i++){
				values[i].move();
			}
			for(i = 0; i<values.length; i++){
				values[i].applyGravity(people);
			}
			for(i = 0; i<values.length; i++){
				values[i].applyRotation(people, -0.1);
			}
		}
		
		public function createNewParticle(emitterID:Number, xPos:Number, yPos:Number):void{
			// emit particles from this point with an initial random vector
			var mass:Number = Math.random();	// 0-1
			var radius:Number = 3 + (Math.random() * 3)
			var particle:Particle = new Particle(particleID, emitterID, xPos, yPos, radius, mass);
			particle.addCallBack(removeParticle);
			particle.setColor(people[emitterID].getParticleColor());
			particleLayer.addChild(particle);
			particles.put(particleID, particle);
			particleID++;
		}
		
		public function removeParticle(e:ParticleEvent){
			// remove particle
			particles.remove(e.id);
			//trace("particle "+ e.id +" removed, "+ particles.size() + " left");
		}
		
	}
	
}