package net.electroland.enteractive.screen {
	
	import flash.display.Sprite;
	import flash.display.MovieClip;
	import flash.events.Event;
	import flash.events.KeyboardEvent;
	import com.ericfeminella.collections.HashMap;
	import org.papervision3d.view.Viewport3D;
	import org.papervision3d.cameras.*;
	import org.papervision3d.scenes.Scene3D;
	import org.papervision3d.render.BasicRenderEngine;
	import org.papervision3d.materials.MovieMaterial;
	import org.papervision3d.objects.primitives.Plane;
	
	/*
	LIGHTMATRIX.as
	by Aaron Siegel, 1-20-2010
	
	This contains the displayable Light objects and receives new
	lighting data from the Java application, then updates the
	Light objects with the new state.
	
	*/
	
	public class LightMatrix extends Sprite{
		
		// light variables
		private var lights:HashMap;
		private var horizontalCount:Number = 18;
		private var verticalCount:Number = 6;
		private var lightWidth = 30;
		private var lightHeight = 30;
		private var verticalSpacing = 50;
		private var horizontalSpacing = 12;
		
		// papervision variables
		public var viewport:Viewport3D;
		public var renderer:BasicRenderEngine;
		public var scene:Scene3D;
		public var camera:Camera3D;
		public var lightPlane:Plane;
		public var translateMode:Number = 0;	// 0 = move camera, 1 = rotate camera, 2 = adjust FOV
		
		public function LightMatrix(){
			lights = new HashMap();
		}
		
		public function init():void{
			// to be called after being added to stage
			initPapervision(1366, 768); // initialise papervision
            init3d(); 					// Initialise the 3d stuff..
			initEvents();
		}
		
		
		private function initPapervision(vpWidth:Number, vpHeight:Number):void {
            // here is where we initialise everything we need to render a papervision scene.
            viewport = new Viewport3D(vpWidth, vpHeight);	// frame for viewing scene
            addChild(viewport); 							// add the viewport to the stage.
            renderer = new BasicRenderEngine();				// render image draws everything
            scene = new Scene3D();							// default scene object
            camera = new Camera3D(53.1);					// default camera object
        }
		
		private function init3d():void {					
			// add light objects here
			var lightsMC:MovieClip = new MovieClip();
			var lightID:Number = 0;
			for(var v:Number = 0; v<verticalCount; v++){
				var ypos:Number = v * (lightHeight + verticalSpacing);
				for(var h:Number = 0; h<horizontalCount; h++){
					var xpos:Number = h * (lightWidth + horizontalSpacing);
					var light:Light = new Light(lightID, xpos, ypos, lightWidth, lightHeight);
					lights.put(lightID, light);
					lightsMC.addChild(light);
					//var mm:MovieMaterial = new MovieMaterial(light, true, true);
					//mm.doubleSided = true;
					//var plane:Plane = new Plane(mm, lightWidth, lightHeight);
					//plane.x = xpos - ((horizontalCount * (lightWidth + horizontalSpacing))/2);
					//plane.y = ((verticalCount * (lightHeight + verticalSpacing))/2) - ypos;
					//scene.addChild(plane);
					lightID++;
				}
			}
			
			// create a large plane to hold all light movieclips in a single parent MC
			var mm:MovieMaterial = new MovieMaterial(lightsMC, true, true, true);
			mm.doubleSided = true;
			lightPlane = new Plane(mm, horizontalCount * (lightWidth + horizontalSpacing), verticalCount * (lightHeight + verticalSpacing));
			scene.addChild(lightPlane);
			
			// SET INITIAL VALUES FOR CAMERA/LIGHTPLANE POSITIONS
			camera.zoom = 13;
			camera.focus = 100;								// sets scaling of MC's to natural state
			camera.x = 238;
			camera.y = -547;
			camera.z = -957;
			camera.rotationX = -29;
			camera.rotationY = -14;
			//camera.rotationZ = 0;
			//camera.orbit(62, 83, true, lightPlane);
		}
		
		private function initEvents():void {
			addEventListener(Event.ENTER_FRAME, frameEvent);
			stage.addEventListener(KeyboardEvent.KEY_DOWN, keyDownEvent);
		}
		
		protected function frameEvent(e:Event):void {
			renderer.renderScene(scene, camera, viewport);
		}
		
		public function keyDownEvent(e:KeyboardEvent):void{
			if(e.keyCode == 37){		// left arrow
				if(translateMode == 0){
					camera.x--;
				} else if(translateMode == 1){
					camera.rotationY--;
				} else if(translateMode == 2){
					
				}
			} else if(e.keyCode == 38){	// up arrow
				if(translateMode == 0){
					camera.y++;
				} else if(translateMode == 1){
					camera.rotationX--;
				} else if(translateMode == 2){
					camera.fov++;
				}
			} else if(e.keyCode == 39){	// right arrow
				if(translateMode == 0){
					camera.x++;
				} else if(translateMode == 1){
					camera.rotationY++;
				} else if(translateMode == 2){
					
				}
			} else if(e.keyCode == 40){	// down arrow
				if(translateMode == 0){
					camera.y--;
				} else if(translateMode == 1){
					camera.rotationX++;
				} else if(translateMode == 2){
					camera.fov--;
				}
			} else if(e.keyCode == 49){ // 1, a.k.a. translate mode
				translateMode = 0;
			} else if(e.keyCode == 50){ // 2, a.k.a. rotate mode
				translateMode = 1;
			} else if(e.keyCode == 51){ // 3, a.k.a. FOV mode
				translateMode = 2;
			} else if(e.keyCode == 80){	// p
				// print out values for camera and lightPlane
				trace("camera zoom: "+ camera.zoom +", x: "+ camera.x + ", y: "+ camera.y + ", z: "+ camera.z);
				trace("camera rotX: "+ camera.rotationX +", rotY: "+ camera.rotationY + ", rotZ: "+ camera.rotationZ);
				trace("camera FOV: "+ camera.fov);
			} else if(e.keyCode == 187){ // plus
				if(translateMode == 0){
					camera.z++;
				} else if(translateMode == 1){
					camera.rotationZ++;
				} else if(translateMode == 2){
					
				}
			} else if(e.keyCode == 189){ // minus
				if(translateMode == 0){
					camera.z--;
				} else if(translateMode == 1){
					camera.rotationZ--;
				} else if(translateMode == 2){
					
				}
			} else {
				trace(e.keyCode);
			}
		}

		
		public function updateLights(xmlData:XML):void{
			// receive lighting data and propogate to light objects
			var lightList:XMLList = xmlData.lights.light;
			for each (var lightElement:XML in lightList){
				//trace(Number(lightElement.toString()));
				var light:Light = lights.getValue(Number(lightElement.attribute("id")));
				light.updateLight(Number(lightElement.toString()));
			}
		}
		
	}
	
}