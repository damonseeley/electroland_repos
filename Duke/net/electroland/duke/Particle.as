package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.geom.ColorTransform;
	import flash.display.DisplayObject;
	import flash.net.URLRequest;
	import flash.events.Event;
	import flash.display.Bitmap;
	import com.greensock.TweenLite;
	import com.greensock.events.TweenEvent;
	import com.greensock.easing.*;
	
	public class Particle extends MovieClip{
		
		public var id:Number;				// unique ID# for particle
		public var emitterID:Number;		// ID# of object that emitted this particle
		public var radius:Number;			// visual size
		public var radiusScale:Number;		// scale value between 0 and 1
		public var minRadius:Number;
		public var maxRadius:Number;
		public var mass:Number;				// multiplier for gravity magnitude
		public var spin:Number;				// amount of rotation per frame in degrees
		public var spinScale:Number;
		public var minSpin:Number;
		public var maxSpin:Number;
		public var xv:Number;				// horizontal vector/velocity
		public var yv:Number;				// vertical vector/velocity
		public var scale:Number;
		private var damping:Number;			// slows velocity over time
		
		private var xdiff:Number;			// used for frame by frame calculation of gravity
		private var ydiff:Number;
		private var hypo:Number;
		private var unitx:Number;
		private var unity:Number;
		private var magnitude:Number;
		
		// star motion variables
		private var massMultiplierMin:Number = 1;
		private var massMultiplierMax:Number = 4;
		private var massMultiplier:Number = 1;
		private var massThrobDuration:Number = 2;
		
		// spring motion variables
		private var springDuration:Number = 2;								// seconds around person
		//public var springPosition:Number = (Math.random() * (Math.PI*2)) - Math.PI;		// initial point in spring action
		public var springPosition:Number = Math.random() * (Math.PI*2)
		public var springDistanceMin:Number = 50;
		public var springDistanceMax:Number = 150;
		public var springDistanceScale:Number = Math.random();
		private var springDistance:Number = (springDistanceScale*(springDistanceMax-springDistanceMin)) + springDistanceMin;	// distance from person obj
		private var springRotation:Number = Math.random() * (Math.PI*2);	// rotation of spring vector
		private var atomicRotationA:Number = Math.random() * (Math.PI);
		private var atomicRotationB:Number = Math.random() * (Math.PI);
		
		
		// visual properties
		private var redColor:Number;		
		private var greenColor:Number;
		private var blueColor:Number;
		private var alphaColor:Number;
		private var visualMode:Number = 0;	// 0 = flash circle, 1 = softParticle.png, 2 = weirdParticle.png, 3 = lineParticle.png
		private var image:Bitmap;
		private var fadingOut:Boolean = false;
		
		// reference to system
		private var particleSystem:ParticleSystem;
		
		/*
		PARTICLE.as
		by Aaron Siegel, 2-1-2010
		
		(1) Each particle is influenced by Person objects.
		*/
		
		public function Particle(id:Number, emitterID:Number, x:Number, y:Number, radiusScale:Number, minRadius:Number, maxRadius:Number,
								 mass:Number, spin:Number, visualMode:Number, particleSystem:ParticleSystem){
			this.id = id;
			this.emitterID = emitterID;
			this.x = x;
			this.y = y;
			this.radiusScale = radiusScale;
			this.minRadius = minRadius;
			this.maxRadius = maxRadius;
			this.radius = (radiusScale * (maxRadius-minRadius)) + minRadius;
			this.mass = mass;
			this.minSpin = particleSystem.people[emitterID].particleSpinMin;
			this.maxSpin = particleSystem.people[emitterID].particleSpinMax;
			this.spin = spin;
			this.spinScale = (spin - minSpin) / (maxSpin - minSpin);
			this.visualMode = visualMode;
			this.particleSystem = particleSystem;
			xv = (Math.random() - 0.5) * 5;
			yv = (Math.random() - 0.5) * 5;
			damping = 0.97;
			
			// draw visual appearance of particle
			this.alpha = 0.8;
			setVisualMode(visualMode);
		}
		
		public function applyGravity(gravityObjects:Array):void{
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 0){
					// get hypotenuse between this particle and the gravity object
					xdiff = x - gravityObjects[i].x;
					ydiff = y - gravityObjects[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					
					if(hypo < gravityObjects[i].radiusOfAttractionMax && hypo > gravityObjects[i].radiusOfAttractionMin){
						// if hypo is less than radiusOfAttractionMax and more than radiusOfAttractionMin
						unitx = xdiff/hypo;						// normalize and create unit vector
						unity = ydiff/hypo;
						magnitude = (1 - (hypo / gravityObjects[i].radiusOfAttractionMax)) * gravityObjects[i].mass;
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
		}
		
		public function applySquareGravity(gravityObjects:Array):void{
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 1){		// if a square gravity producer...
					xdiff = x - gravityObjects[i].x;
					ydiff = y - gravityObjects[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					if(hypo < gravityObjects[i].radiusOfAttractionMax){		// if hypo is less than radiusOfAttractionMax...
						// pull towards gravity object
						unitx = xdiff/hypo;						// normalize and create unit vector
						unity = ydiff/hypo;
						magnitude = (1 - (hypo / gravityObjects[i].radiusOfAttractionMax)) * gravityObjects[i].mass;
						xv -= unitx * magnitude;
						yv -= unity * magnitude;
					}
					if(Math.abs(xdiff) < gravityObjects[i].radiusOfRepulsion && Math.abs(ydiff) < gravityObjects[i].radiusOfRepulsion){
						// if particle inside box, FORCE it immediately out to the edge along its current unit vector
						unitx = xdiff/hypo;						// normalize and create unit vector
						unity = ydiff/hypo;
						//magnitude = (1 - (hypo / gravityObjects[i].radiusOfRepulsion)) * gravityObjects[i].mass;
						//x = gravityObjects[i].x + (gravityObjects[i].radiusOfRepulsion * unitx);		// THIS IS ROUND
						//y = gravityObjects[i].y + (gravityObjects[i].radiusOfRepulsion * unity);
						var boxhypo:Number = Math.sqrt(gravityObjects[i].radiusOfRepulsion*gravityObjects[i].radiusOfRepulsion + gravityObjects[i].radiusOfRepulsion*gravityObjects[i].radiusOfRepulsion);
						if(Math.abs(xdiff) > Math.abs(ydiff)){	// if xdiff is longer distance than ydiff
							// push x position out until it is equal to radiusOfRepulsion
							if(unitx > 0){
								x = gravityObjects[i].x + gravityObjects[i].radiusOfRepulsion;
							} else {
								x = gravityObjects[i].x - gravityObjects[i].radiusOfRepulsion;
							}
							if(unity > 0){
								//y = gravityObjects[i].y + (boxhypo * unity);
							} else {
								//y = gravityObjects[i].y - (boxhypo * unity);
							}
						} else {								// if ydiff is longer distance than xdiff
							// push y position out until it is equal to radiusOfRepulsion
							if(unity > 0){
								y = gravityObjects[i].y + gravityObjects[i].radiusOfRepulsion;
							} else {
								y = gravityObjects[i].y - gravityObjects[i].radiusOfRepulsion;
							}
							if(unitx > 0){
								//x = gravityObjects[i].x + (boxhypo * unitx);
							} else {
								//x = gravityObjects[i].x - (boxhypo * unitx);
							}
						}
					}
				}
			}
		}
		
		public function applyStarGravity(gravityObjects:Array):void{	// cause particle to bounce around person in star shape
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 2){		// if a star gravity producer...
					// this should behave similarly to the regular gravity system except...
					// mass multiplier should fluctuate with time to cause particle to dip to and from the gravity object.
					// get hypotenuse between this particle and the gravity object
					xdiff = x - gravityObjects[i].x;
					ydiff = y - gravityObjects[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					
					if(hypo < gravityObjects[i].radiusOfAttractionMax && hypo > gravityObjects[i].radiusOfAttractionMin){
						// if hypo is less than radiusOfAttractionMax and more than radiusOfAttractionMin
						unitx = xdiff/hypo;						// normalize and create unit vector
						unity = ydiff/hypo;
						magnitude = (1 - (hypo / gravityObjects[i].radiusOfAttractionMax)) * gravityObjects[i].mass;
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
		}
		
		public function applySpringGravity(gravityObjects:Array):void{
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 3){			// if spring gravity...
					xv = 0;
					yv = 0;
					//x = (springDistance * Math.cos(springPosition)) + particleSystem.people[emitterID].x;
					//y = (springDistance * Math.sin(springPosition)) + particleSystem.people[emitterID].y;
					//y = particleSystem.people[emitterID].y;
					
					var pos:Number = (springDistance * Math.cos(springPosition));
					var rotX:Number = Math.cos(springRotation)*pos;
					var rotY:Number = Math.sin(springRotation)*pos;
					x = rotX + particleSystem.people[emitterID].x;
					y = rotY + particleSystem.people[emitterID].y;
					
				}
			}
		}
		
		public function applyAtomicGravity(gravityObjects:Array):void{
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 4){			// if atomic gravity...
					xv = 0;
					yv = 0;
					var xpos:Number = (springDistanceMax * Math.cos(springPosition));
					var ypos:Number = (springDistanceMax * Math.sin(springPosition));
					var rotX:Number = Math.cos(springRotation)*xpos;
					var rotY:Number = Math.sin(springRotation)*ypos;
					x = rotX + particleSystem.people[emitterID].x;
					y = rotY + particleSystem.people[emitterID].y;
				}
			}
		}
		
		public function applyRotation(gravityObjects:Array):void{
			// torque is a magnifier for movement perpindicular to line between gravity object and this particle.
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 0 || gravityObjects[i].gravityMode == 2){
					// get hypotenuse between this particle and the gravity object
					xdiff = x - gravityObjects[i].x;
					ydiff = y - gravityObjects[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					
					if(hypo < gravityObjects[i].radiusOfRotation){
						unitx = xdiff/hypo;						// normalize and create unit vector
						unity = ydiff/hypo;
						// the closer the particle, the faster it rotates
						magnitude = (1 - (hypo / gravityObjects[i].radiusOfRotation)) * gravityObjects[i].torque * mass;
						xv -= unity * magnitude;	// swap vectors to go perpindicular
						yv += unitx * magnitude;
						// measure the new hypo between particle and person
						var newxdiff:Number = (x+xv) - gravityObjects[i].x;
						var newydiff:Number = (y+yv) - gravityObjects[i].y;
						var newhypo:Number = Math.sqrt(newxdiff*newxdiff + newydiff*newydiff);
						// get difference between new and old hypo
						var hypodiff:Number = hypo - newhypo;
						// move particle along new unit vector to reduce hypo by difference
						//x += (newxdiff/newhypo) * hypodiff;
						//y += (newydiff/newhypo) * hypodiff;
						//xv += (hypodiff/newhypo) * newxdiff;	// THIS IS WEIRD
						//yv += (hypodiff/newhypo) * newydiff;
					}
				}
			}
		}
		
		public function applySquareRotation(gravityObjects:Array):void{
			// torque is a magnifier for movement perpindicular to line between gravity object and this particle.
			for(var i:Number = 0; i<gravityObjects.length; i++){
				if(gravityObjects[i].gravityMode == 1){		// if a square gravity producer...
					xdiff = x - gravityObjects[i].x;
					ydiff = y - gravityObjects[i].y;
					hypo = Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					if(hypo < gravityObjects[i].radiusOfRotation){
						// the closer the particle, the faster it rotates
						//magnitude = (1 - (hypo / gravityObjects[i].radiusOfRotation)) * gravityObjects[i].torque * mass;
						magnitude = gravityObjects[i].torque;
						if(Math.abs(xdiff) > Math.abs(ydiff)){	// if further to the side than to the top/bottom
							if(xdiff > 0){						// if particle is to the right of person...
								yv -= magnitude;				// move particle up
							} else {							// if particle is to the left...
								yv += magnitude;				// move particle down
							}
							xv = 0;
						} else {								// if further to the top/bottom than to the side
							if(ydiff > 0){						// if particle is below person...
								xv += magnitude;				// move particle right
							} else {							// if particle is above person...
								xv -= magnitude;				// move particle left
							}
							yv = 0;
						}
					}
				}
			}
		}
		
		public function die():void{
			//trace("particle "+id+" die!");
			particleSystem.removeParticle(new ParticleEvent(id));
		}
		
		public function throbMassOut():void{
			TweenLite.to(this, massThrobDuration, {massMultiplier:massMultiplierMax, onComplete:throbMassIn});
		}
		
		public function throbMassIn():void{
			TweenLite.to(this, massThrobDuration, {massMultiplier:massMultiplierMin, onComplete:throbMassOut});
		}
		
		public function move():void{
			// if velocity drops to approximately zero, begin fade out and die procedure.
			if(particleSystem.people[emitterID].gravityMode == 0){
				if(Math.abs(xv) < 0.01 && Math.abs(yv) < 0.01){
					if(!fadingOut){
						fadingOut = true;
						TweenLite.to(this, 5, {alpha:0, onComplete:die});
					}
				}
			}
			x += xv;
			y += yv;
			xv *= damping;
			yv *= damping;
			this.rotation += spin;
		}
		
		public function setVisualMode(visualMode:Number):void{
			//trace(visualMode);
			this.visualMode = visualMode;
			var bm:Bitmap;
			if(visualMode == 0){
				
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				this.graphics.beginFill(0x0099FF);	// must convert from RGB integers
				this.graphics.drawCircle(0, 0, radius);
				this.graphics.endFill();
				
				/*
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				
				bm = particleSystem.redDot.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				*/
			} else if(visualMode == 1){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.greenDot.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor();
			} else if(visualMode == 2){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.blueDot.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor()
			} else if(visualMode == 3){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.hexagon.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor()
			} else if(visualMode == 4){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.cross.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor()
			} else if(visualMode == 5){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.lineParticle.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor()
			} else if(visualMode == 6){
				this.graphics.clear();
				if(image != null && this.contains(image)){
					removeChild(image);
				}
				bm = particleSystem.roundRect.content as Bitmap;
				image = new Bitmap(bm.bitmapData);
				image.width = radius*2;
				image.height = radius*2;
				addChild(image);
				image.x = 0 - radius;
				image.y = 0 - radius;
				clearColor()
			}
			//setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		public function imageLoaded(e:Event):void{
			//trace("image loaded");
			image.width = radius*2;
			image.height = radius*2;
		}
		
		
		public function clearColor():void{
			this.transform.colorTransform = new ColorTransform();
		}
		
		public function setColor(colors:Array):void{
			redColor = colors[0];	// for storage
			greenColor = colors[1];
			blueColor = colors[2];
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
			var ct:ColorTransform = this.transform.colorTransform;
			ct.color = uint("0x"+redHex+greenHex+blueHex);
			this.transform.colorTransform = ct;
			this.alpha = colors[3];
		}
		
		public function setRed(red:Number):void{
			redColor = red;
			setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		public function setGreen(green:Number):void{
			greenColor = green;
			setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		public function setBlue(blue:Number):void{
			blueColor = blue;
			setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		public function setSpin(spin:Number):void{
			this.spin = spin;
		}
		
		public function setMinSpin(minSpin:Number):void{
			this.minSpin = minSpin;
			spin = (spinScale * (maxSpin-minSpin)) + minSpin;
		}
		
		public function setMaxSpin(maxSpin:Number):void{
			this.maxSpin = maxSpin;
			spin = (spinScale * (maxSpin-minSpin)) + minSpin;
		}
		
		public function setMinRadius(minRadius:Number):void{
			this.minRadius = minRadius;
			radius = (radiusScale * (maxRadius-minRadius)) + minRadius;

			if(visualMode == 0){
				this.graphics.clear();
				this.graphics.beginFill(0x0099FF);	// must convert from RGB integers
				this.graphics.drawCircle(0, 0, radius);
				this.graphics.endFill();
			} else {
				image.width = radius*2;
				image.height = radius*2;
				image.x = 0 - radius;
				image.y = 0 - radius;
			}
			//setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		public function setMaxRadius(maxRadius:Number):void{
			this.maxRadius = maxRadius;
			this.radius = (radiusScale * (maxRadius-minRadius)) + minRadius;
			
			if(visualMode == 0){
				this.graphics.clear();
				this.graphics.beginFill(0x0099FF);	// must convert from RGB integers
				this.graphics.drawCircle(0, 0, radius);
				this.graphics.endFill();
			} else {
				image.width = radius*2;
				image.height = radius*2;
				image.x = 0 - radius;
				image.y = 0 - radius;
			}
			//setColor([redColor, greenColor, blueColor, alpha]);
		}
		
		
		
		
		
		
		// forced animations
		
		public function beginSpring():void{
			var duration:Number = springDuration - (springDuration * (springPosition / (2*Math.PI)));
			TweenLite.to(this, duration, {springPosition:Math.PI*2, ease:Linear.easeNone, onComplete:repeatSpring});	// looping tween
		}
		
		public function repeatSpring():void{
			springPosition = 0;
			beginSpring();
		}
	}
	
}