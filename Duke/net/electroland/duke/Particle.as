package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.geom.ColorTransform;
	import com.greensock.TweenLite;
	import com.greensock.events.TweenEvent;
	
	public class Particle extends MovieClip{
		
		public var id:Number;				// unique ID# for particle
		public var emitterID:Number;		// ID# of object that emitted this particle
		public var radius:Number;			// visual size
		public var mass:Number;				// multiplier for gravity magnitude
		public var xv:Number;				// horizontal vector/velocity
		public var yv:Number;				// vertical vector/velocity
		private var damping:Number;			// slows velocity over time
		
		private var xdiff:Number;			// used for frame by frame calculation of gravity
		private var ydiff:Number;
		private var hypo:Number;
		private var unitx:Number;
		private var unity:Number;
		private var magnitude:Number;
		
		private var redColor:Number;		// visual properties
		private var greenColor:Number;
		private var blueColor:Number;
		private var alphaColor:Number;
		
		private var fadeOutTween:TweenLite;
		private var particleSystem:ParticleSystem;
		
		/*
		PARTICLE.as
		by Aaron Siegel, 2-1-2010
		
		(1) Each particle is influenced by either Person objects, other particles, or other sources of gravity.
		*/
		
		public function Particle(id:Number, emitterID:Number, x:Number, y:Number, radius:Number, mass:Number){
			this.id = id;
			this.emitterID = emitterID;
			this.x = x;
			this.y = y;
			this.radius = radius;
			this.mass = mass;
			xv = (Math.random() - 0.5) * 5;
			yv = (Math.random() - 0.5) * 5;
			damping = 0.97;
			
			// draw visual appearance of particle
			this.graphics.beginFill(0x0099FF);	// must convert from RGB integers
			this.graphics.drawCircle(0, 0, radius);
			this.graphics.endFill();
			this.alpha = 0.8;
		}
		
		public function applyGravity(gravityObjects:Array):void{
			for(var i:Number = 0; i<gravityObjects.length; i++){
				// get hypotenuse between this particle and the gravity object
				xdiff = x - gravityObjects[i].x;
				ydiff = y - gravityObjects[i].y;
				hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
				
				if(hypo < gravityObjects[i].radiusOfAttraction && hypo > gravityObjects[i].radiusOfRepulsion + 50){		// if hypo is less than radiusOfAttraction
					unitx = xdiff/hypo;						// normalize and create unit vector
					unity = ydiff/hypo;
					magnitude = (1 - (hypo / gravityObjects[i].radiusOfAttraction)) * gravityObjects[i].mass;
					xv -= unitx * magnitude;
					yv -= unity * magnitude;
				}
				if(hypo < gravityObjects[i].radiusOfRepulsion){			// if hypo is less than radiusOfRepulsion
					unitx = xdiff/hypo;						// normalize and create unit vector
					unity = ydiff/hypo;
					magnitude = (1 - (hypo / gravityObjects[i].radiusOfRepulsion)) * gravityObjects[i].mass;
					xv += unitx * magnitude;
					yv += unity * magnitude;
				}
				
			}
		}
		
		public function applyRotation(gravityObjects:Array, torque:Number):void{
			// torque is a magnifier for movement perpindicular to line between gravity object and this particle.
			for(var i:Number = 0; i<gravityObjects.length; i++){
				// get hypotenuse between this particle and the gravity object
				xdiff = x - gravityObjects[i].x;
				ydiff = y - gravityObjects[i].y;
				hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
				
				if(hypo < gravityObjects[i].radiusOfRotation){
					unitx = xdiff/hypo;						// normalize and create unit vector
					unity = ydiff/hypo;
					// the closer the particle, the faster it rotates
					magnitude = (1 - (hypo / gravityObjects[i].radiusOfRotation)) * torque * mass;
					xv -= unity * magnitude;	// swap vectors to go perpindicular
					yv += unitx * magnitude;
				}
				
			}
		}
		
		public function addCallBack(particleSystem:ParticleSystem):void{
			this.particleSystem = particleSystem;
		}
		
		public function die():void{
			//deathCallBack(new ParticleEvent(id));
			particleSystem.removeParticle(new ParticleEvent(id));
		}
		
		public function move():void{
			// if velocity drops to approximately zero, begin fade out and die procedure.
			// TODO: add a timer that can be reset to require movement to be null for a period before dying.
			if(Math.abs(xv) < 0.01 && Math.abs(yv) < 0.01){
				TweenLite.to(this, 5, {alpha:0, onComplete:die});
			}
			x += xv;
			y += yv;
			xv *= damping;
			yv *= damping;
		}
		
		public function setColor(colors:Array){
			var ct:ColorTransform = this.transform.colorTransform;
			//ct.redOffset(colors[0]);
			//ct.greenOffset(colors[1]);
			//ct.blueOffset(colors[2]);
			var redHex:String = colors[0].toString(16);
			if(redHex.length < 2){
				redHex = "0"+redHex;
			}
			var greenHex:String = colors[1].toString(16);
			if(greenHex.length < 2){
				greenHex = "0"+greenHex;
			}
			var blueHex:String = colors[2].toString(16);
			if(blueHex.length < 2){
				blueHex = "0"+blueHex;
			}
			//trace("0x"+redHex+greenHex+blueHex);
			ct.color = uint("0x"+redHex+greenHex+blueHex);
			this.transform.colorTransform = ct;
			this.alpha = colors[3];
		}
		
	}
	
}