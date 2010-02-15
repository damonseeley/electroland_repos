﻿package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	import flash.events.Event;
	import flash.events.KeyboardEvent;
	import flash.events.TimerEvent;
	import flash.utils.Timer;
	import flash.geom.ColorTransform;
	
	public class Person extends MovieClip{
		
		public var id:Number;						// unique ID# of person
		public var radius:Number;					// visual radius
		public var radiusOfAttractionMax:Number;	// must be atleast this close to be pulled towards person
		public var radiusOfAttractionMin:Number;	// must be atleast this far away to be pulled towards person
		public var radiusOfRepulsion:Number;		// must be atleast this close to be pushed away from person
		public var radiusOfRotation:Number;			// must be atleast this close to be rotated around a person
		public var mass:Number;						// multiplier to increase gravity magnitude
		public var torque:Number;					// rotational force applied to particles
		
		public var personColorRed:Number;			// visual properties of person object
		public var personColorGreen:Number;
		public var personColorBlue:Number;
		public var personColorAlpha:Number;
		public var particleColorRed:Number;			// visual properties of particle objects
		public var particleColorGreen:Number;
		public var particleColorBlue:Number;
		public var particleColorAlpha:Number;
		
		public var selected:Boolean = false;
		private var particleSystem:ParticleSystem;
		private var distanceBetweenParticles:Number;
		private var lastParticleX:Number;
		private var lastParticleY:Number;
		public var particleMinRadius = 3;
		public var particleMaxRadius = 6;
		public var particleSpinMin;
		public var particleSpinMax;
		public var particleCount = 20;				// regulated number of particles emitted from this person
		public var springSpeed = 2;
		public var atomicSpeed = 2;
		
		public var sparkLifeMin:Number = 1;
		public var sparkLifeMax:Number = 2;
		public var sparkSpeed:Number = 20;
		public var emitterSpeed:Number = 50;		// delay between particles in milliseconds
		
		public var gravityMode = 0;					// 0 = regular gravity, 1 = square gravity
		public var visualMode = 0;					// visual mode of particles emitted from this person
		public var remove:Boolean = false;			// whether removeNow should be triggered when clicked
		public var removeNow:Boolean = false;		// tells particle system to remove at end of loop
		
		/*
		PERSON.as
		by Aaron Siegel, 2-1-2010
		
		Person objects are attractors, detractors, and particle emitters.
		(1) Person's have influence over Particle objects but not over other Person objects.
		(2) Person objects occasionally abandon particles when they go out of range, leaving a trail behind a moving Person.
		(3) Person objects also emit particles as they move to replace the ones they dispense.
		(4) Person objects can attract particles emitted from other Person objects.
		*/
		
		public function Person(id:Number, x:Number, y:Number, radius:Number, mass:Number, torque:Number){
			this.id = id;
			this.x = x;
			this.y = y;
			this.radius = radius;
			this.radiusOfRepulsion = radius*2;
			this.radiusOfAttractionMax = radius*10;
			this.radiusOfAttractionMin = radius*2 + 50;
			this.radiusOfRotation = radius*10;
			this.mass = mass;
			this.torque = torque;
			
			// properties regarding creation of new particles
			distanceBetweenParticles = radius;	// amount this must move before creating a new particle
			lastParticleX = x;
			lastParticleY = y;
			particleSpinMin = -2;	// degrees per frame
			particleSpinMax = 2;
			
			// draw visual appearance of person object
			this.graphics.beginFill(0xAAAAAAAA);
			this.graphics.drawCircle(0, 0, radius);
			this.graphics.endFill();
			this.alpha = 0.5;
			
			// add event listeners for user interaction
			this.addEventListener(MouseEvent.MOUSE_DOWN, onMouseDown);
			this.addEventListener(MouseEvent.MOUSE_UP, onMouseUp);
			this.addEventListener(Event.ENTER_FRAME, onEnterFrame);
		}
		
		public function getParticleColor():Array{
			var colors:Array = new Array();
			colors.push(particleColorRed);
			colors.push(particleColorGreen);
			colors.push(particleColorBlue);
			colors.push(particleColorAlpha);
			return colors;
		}
		
		public function setParticleColor(colors:Array):void{
			particleColorRed = colors[0];
			particleColorGreen = colors[1];
			particleColorBlue = colors[2];
			particleColorAlpha = colors[3];
		}
		
		public function addCallback(particleSystem:ParticleSystem):void{
			// notify particle system when a person has been selected
			this.particleSystem = particleSystem;
			stage.addEventListener(KeyboardEvent.KEY_DOWN,keyDownListener);
			stage.addEventListener(KeyboardEvent.KEY_UP,keyUpListener);
		}
		
		public function onMouseDown(event:MouseEvent):void{
			if(remove){
				//particleSystem.people.remove(id);
				particleSystem.removePerson(id);
				//removeNow = true;	// removed by particle system
			} else {
				startDrag();
				select();
			}
		}
		
		public function onMouseUp(event:MouseEvent):void{
			stopDrag();
		}
		
		public function keyDownListener(e:KeyboardEvent):void{
			if(e.ctrlKey){
				remove = true;
			}
		}
		
		public function keyUpListener(e:KeyboardEvent):void{
			remove = false;
		}
		
		public function onEnterFrame(event:Event):void{
			/*
			var xdiff:Number = x - lastParticleX;
			var ydiff:Number = y - lastParticleY;
			var hypo:Number = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
			if(gravityMode <= 1){
				if(hypo > distanceBetweenParticles){
					lastParticleX = x;
					lastParticleY = y;
					var spin:Number = particleSpinMin + (Math.random() * (particleSpinMax - particleSpinMin));
					particleSystem.createNewParticle(id, x, y, spin, visualMode);
				}
			}
			*/
		}
		
		public function select():void{
			var ct:ColorTransform = this.transform.colorTransform;
			ct.color = 0x666666;
			this.transform.colorTransform = ct;
			selected = true;
			particleSystem.personSelected(id);
		}
		
		public function deselect():void{
			var ct:ColorTransform = this.transform.colorTransform;
			ct.color = 0xAAAAAA;
			this.transform.colorTransform = ct;
			selected = false;
		}
		
		public function silentSelect():void{
			var ct:ColorTransform = this.transform.colorTransform;
			ct.color = 0x666666;
			this.transform.colorTransform = ct;
			selected = true;
		}
		
		// FUNCTIONS TO CHANGE PROPERTIES ON THE FLY FROM CONTROL PANEL
		
		public function setRadiusOfAttractionMax(radiusOfAttractionMax:Number):void{
			this.radiusOfAttractionMax = radiusOfAttractionMax;
		}
		
		public function setRadiusOfAttractionMin(radiusOfAttractionMin:Number):void{
			this.radiusOfAttractionMin = radiusOfAttractionMin;
		}
		
		public function setRadiusOfRepulsion(radiusOfRepulsion:Number):void{
			this.radiusOfRepulsion = radiusOfRepulsion;
		} 
		
		public function setTorque(torque:Number):void{
			this.torque = torque;
		}
		
		public function setMass(mass:Number):void{
			this.mass = mass;
		}
		
		public function setRed(particleColorRed:Number):void{
			this.particleColorRed = particleColorRed;
		}
		
		public function setGreen(particleColorGreen:Number):void{
			this.particleColorGreen = particleColorGreen;
		}
		
		public function setBlue(particleColorBlue:Number):void{
			this.particleColorBlue = particleColorBlue;
		}
		
		public function setParticleMinSize(particleMinRadius:Number):void{
			this.particleMinRadius = particleMinRadius;
		}
		
		public function setParticleMaxSize(particleMaxRadius:Number):void{
			this.particleMaxRadius = particleMaxRadius;
		}
		
		public function setParticleMinSpin(particleSpinMin:Number):void{
			this.particleSpinMin = particleSpinMin;
		}
		
		public function setParticleMaxSpin(particleSpinMax:Number):void{
			this.particleSpinMax = particleSpinMax;
		}
		
		public function setParticleCount(particleCount:Number):void{
			this.particleCount = particleCount;
		}
		
		public function setVisualMode(visualMode:Number):void{
			this.visualMode = visualMode;
		}
		
		public function setGravityMode(gravityMode:Number):void{
			//trace(id +" gravity mode "+gravityMode);
			this.gravityMode = gravityMode;
			var xdiff:Number;
			var ydiff:Number;
			var hypo:Number;
			var values:Array = particleSystem.particles.getValues();
			if(gravityMode == 3){	// spring mode
				for(var i:Number = 0; i<values.length; i++){
					xdiff = x - values[i].x;
					ydiff = y - values[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					if(hypo < radiusOfAttractionMax){
						values[i].beginSpring();
					}
				}
			} else if (gravityMode == 4){	// atomic mode
				for(i = 0; i<values.length; i++){
					xdiff = x - values[i].x;
					ydiff = y - values[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					if(hypo < radiusOfAttractionMax){
						values[i].beginEllipse();
					}
				}
			} else if(gravityMode == 5){	// spark mode
				emitParticle();
			}
		}
		
		public function setSparkLifeMin(sparkLifeMin:Number):void{
			this.sparkLifeMin = sparkLifeMin;
		}
		
		public function setSparkLifeMax(sparkLifeMax:Number):void{
			this.sparkLifeMax = sparkLifeMax;
		}
		
		public function setSparkSpeed(sparkSpeed:Number):void{
			this.sparkSpeed = sparkSpeed;
		}
		
		public function setSparkEmitterDelay(emitterSpeed:Number):void{
			this.emitterSpeed = emitterSpeed;
		}
		
		public function setSpringSpeed(springSpeed:Number):void{
			this.springSpeed = springSpeed;
		}
		
		public function setAtomicSpeed(atomicSpeed:Number):void{
			this.atomicSpeed = atomicSpeed;
		}
		
		public function emitParticle():void{
			//trace("emit particle");
			var spin:Number = (Math.random() * (particleSpinMax - particleSpinMin)) + particleSpinMin;
			particleSystem.createNewParticle(id, x, y, spin, visualMode);
			var timer:Timer = new Timer(emitterSpeed, 1);
			timer.addEventListener("timer", repeatEmitter);
			timer.start();
			//trace(timer.delay);
		}
		
		public function repeatEmitter(e:TimerEvent):void{
			if(gravityMode == 5){
				//trace("repeating");
				emitParticle();
			}
		}
	}
	
}