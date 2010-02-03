package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	
	public class Person extends MovieClip{
		
		public var id:Number;						// unique ID# of person
		public var radius:Number;					// visual radius
		public var radiusOfAttraction:Number;		// must be atleast this close to be pulled towards person
		public var radiusOfRepulsion:Number;		// must be atleast this close to be pushed away from person
		public var radiusOfRotation:Number;			// must be atleast this close to be rotated around a person
		public var mass:Number;						// multiplier to increase gravity magnitude
		
		public var personColorRed:Number;			// visual properties of person object
		public var personColorGreen:Number;
		public var personColorBlue:Number;
		public var personColorAlpha:Number;
		public var particleColorRed:Number;			// visual properties of particle objects
		public var particleColorGreen:Number;
		public var particleColorBlue:Number;
		public var particleColorAlpha:Number;
		
		/*
		PERSON.as
		by Aaron Siegel, 2-1-2010
		
		Person objects are attractors, detractors, and particle emitters.
		(1) Person's have influence over Particle objects but not over other Person objects.
		(2) Person objects occasionally abandon particles when they go out of range, leaving a trail behind a moving Person.
		(3) Person objects also emit particles as they move to replace the ones they dispense.
		(4) Person objects can attract particles emitted from other Person objects.
		*/
		
		public function Person(id:Number, x:Number, y:Number, radius:Number, mass:Number){
			this.id = id;
			this.x = x;
			this.y = y;
			this.radius = radius;
			this.radiusOfRepulsion = radius*2;
			this.radiusOfAttraction = radius*10;
			this.radiusOfRotation = radius*10;
			this.mass = mass;
			
			// draw visual appearance of person object
			this.graphics.beginFill(0xAAAAAAAA);
			this.graphics.drawCircle(0, 0, radius);
			this.graphics.endFill();
			this.alpha = 0.5;
			
			// add event listeners for user interaction
			this.addEventListener(MouseEvent.MOUSE_DOWN, onMouseDown);
			this.addEventListener(MouseEvent.MOUSE_UP, onMouseUp);
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
		
		public function onMouseDown(event:MouseEvent):void{
			startDrag();
		}
		
		public function onMouseUp(event:MouseEvent):void{
			stopDrag();
		}
		
	}
	
}