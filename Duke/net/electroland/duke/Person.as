package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	import flash.events.Event;
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
		}
		
		public function onMouseDown(event:MouseEvent):void{
			startDrag();
			select();
		}
		
		public function onMouseUp(event:MouseEvent):void{
			stopDrag();
		}
		
		public function onEnterFrame(event:Event):void{
			var xdiff:Number = x - lastParticleX;
			var ydiff:Number = y - lastParticleY;
			var hypo:Number = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
			if(hypo > distanceBetweenParticles){
				lastParticleX = x;
				lastParticleY = y;
				particleSystem.createNewParticle(id, x, y);
			}
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
		
	}
	
}