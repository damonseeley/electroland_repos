package net.electroland.enteractive.screen {
	
	import flash.display.Sprite;
	import flash.display.MovieClip;
	import flash.events.Event;
	import flash.events.MouseEvent;
	import flash.events.KeyboardEvent;
	import com.ericfeminella.collections.HashMap;
	import org.papervision3d.view.Viewport3D;
	import org.papervision3d.cameras.*;
	import org.papervision3d.scenes.Scene3D;
	import org.papervision3d.render.BasicRenderEngine;
	//import org.papervision3d.materials.ColorMaterial;
	import org.papervision3d.materials.MovieMaterial;
	//import org.papervision3d.materials.MovieAssetMaterial;
	//import org.papervision3d.materials.WireframeMaterial;
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
		public var dragging:Boolean = false;
		
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
			//camera = new DebugCamera3D(viewport, 53.1);
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
			//var mm:ColorMaterial = new ColorMaterial(0xff0000, 0.5);
			//var mm:WireframeMaterial = new WireframeMaterial(0xff0000, 0.5);
			mm.doubleSided = true;
			lightPlane = new Plane(mm, horizontalCount * (lightWidth + horizontalSpacing), verticalCount * (lightHeight + verticalSpacing));
			//lightPlane.x = -30;
			//lightPlane.y = 33;
			scene.addChild(lightPlane);
			
			// SET INITIAL VALUES FOR CAMERA/LIGHTPLANE POSITIONS
			camera.zoom = 11;
			camera.focus = 100;								// sets scaling of MC's to natural state
			//camera.orbit(62, 83, true, lightPlane);
		}
		
		private function initEvents():void {
			addEventListener(Event.ENTER_FRAME, frameEvent);
			stage.addEventListener(MouseEvent.MOUSE_DOWN, mouseDownEvent);
			stage.addEventListener(MouseEvent.MOUSE_UP, mouseUpEvent);			
			stage.addEventListener(MouseEvent.MOUSE_WHEEL, mouseWheelEvent);
			stage.addEventListener(KeyboardEvent.KEY_DOWN, keyDownEvent);
		}
		
		protected function frameEvent(e:Event):void {
			if(dragging){
				//lightPlane.rotationX = mouseY;
				//lightPlane.rotationY = mouseX;
				//camera.orbit(mouseY, mouseX, true, lightPlane);
				camera.orbit(mouseY, mouseX);
			}
			renderer.renderScene(scene, camera, viewport);
		}
		
		public function mouseDownEvent(e:MouseEvent):void{
			dragging = true;
		}
		
		public function mouseUpEvent(e:MouseEvent):void{
			dragging = false;
		}
		
		public function mouseWheelEvent(e:MouseEvent):void{
			camera.zoom += e.delta * 0.1;
		}
		
		public function keyDownEvent(e:KeyboardEvent):void{
			if(e.keyCode == 37){		// left
				lightPlane.x -= 1;
			} else if(e.keyCode == 38){	// up
				lightPlane.y += 1;
			} else if(e.keyCode == 39){	// right
				lightPlane.x += 1;
			} else if(e.keyCode == 40){	// down
				lightPlane.y -= 1;
			} else if(e.keyCode == 80){	// p
				// TODO: print out values for camera and lightPlane
				trace("lightPlane.x: "+ lightPlane.x +", lightPlane.y: "+ lightPlane.y);
				trace("camera zoom: "+ camera.zoom +", pitch: "+ camera.rotationY + ", yaw: "+ camera.rotationX);
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