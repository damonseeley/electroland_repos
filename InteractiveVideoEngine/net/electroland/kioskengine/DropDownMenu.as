package net.electroland.kioskengine {
	
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.*;
	
	import net.electroland.utils.*;
		
	public class DropDownMenu extends Sprite{
		
		private var conductor:Conductor;
		private var menuItems:Array = new Array();	// holds sprites used as menu items
		private var menuName:String;
		private var textFormat:TextFormat;
		private var titleField:TextField;
		private var menuBG:Sprite;
		
		/*
		
		DROPDOWNMENU.as
		by Aaron Siegel, 7-24-09
		
		Used to select a module, or unit within the current module, to go to.
		
		*/
		
		public function DropDownMenu(menuName:String){
			this.menuName = menuName;
			textFormat = new TextFormat();
			textFormat.font = "Arial";
			textFormat.size = 10;
			titleField = new TextField();
			titleField.text = menuName +":";
			titleField.selectable = false;
			titleField.textColor = 0x000000;
			titleField.blendMode = BlendMode.LAYER;
			titleField.background = true;
			titleField.autoSize = TextFieldAutoSize.LEFT;
			titleField.setTextFormat(textFormat);
			
			addChild(titleField);
			
			this.addEventListener(MouseEvent.MOUSE_OVER, mainMouseOver);
			this.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
		}
		
		public function addConductor(conductor:Conductor){
			this.conductor = conductor;
		}
		
		public function updateItems(itemNames:Array){
			menuBG = new Sprite();
			menuItems = new Array();
			var itemOffset:Number = 14;
			
			for (var i=0; i<itemNames.length; i++){
				var itemEntry:Sprite = new Sprite();
				var itemLabel:TextField = new TextField();
				itemLabel.text = itemNames[i];
				itemLabel.selectable = false;
				itemLabel.blendMode = BlendMode.LAYER;
				itemLabel.background = true;
				itemLabel.autoSize = TextFieldAutoSize.LEFT;
				itemLabel.setTextFormat(textFormat);
				itemEntry.y = itemOffset*i*1;
				itemEntry.buttonMode = true;
				itemEntry.addChild(itemLabel);
				
				menuItems.push(itemEntry);
				
				menuBG.y = itemOffset;
				menuBG.addChild(itemEntry);
				menuBG.visible = false;

			}
			
			menuBG.addChildAt(ELGfx.filledRect(0,0,menuBG.width,menuBG.height,0xFFFFFF,0), 0);
			menuBG.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			this.addChild(menuBG);
		}
		
		private function addItemEvents():void {
			this.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			for each (var i:Sprite in menuItems){
				i.addEventListener(MouseEvent.MOUSE_UP, mousePressed);
			}
		}
		
		private function removeItemEvents():void {
			this.removeEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			for each (var i:Sprite in menuItems){
				i.removeEventListener(MouseEvent.MOUSE_UP, mousePressed);
			}
		}
		
		private function hideMenu():void {
			removeItemEvents();
			menuBG.visible = false;
		}
		
		private function showMenu():void {
			addItemEvents();
			menuBG.visible = true;
		}
		
		private function mainMouseOver(event:MouseEvent):void{
			showMenu();
			if (event.target is TextField) {
				var tf:TextFormat = event.target.getTextFormat();
				tf.bold = true;
				event.target.setTextFormat(tf);
				event.target.textColor = 0xaa0000;
			}
		}
		
		private function mainMouseOut(event:MouseEvent):void{
			hideMenu();
			if (event.target is TextField) {
				var tf:TextFormat = event.target.getTextFormat();
				tf.bold = false;
				event.target.setTextFormat(tf);
				event.target.textColor = 0x000000;
			}
		}
		
		public function mousePressed(event:MouseEvent):void{
			// make the menu items invisible
			hideMenu();
			
			trace(menuName+": "+event.target.text+" menu item clicked");
			
			if(menuName == "MODULES"){
				conductor.jumpToModule(event.target.text);
			} else if(menuName == "UNITS"){
				conductor.jumpToUnit(event.target.text);
			}
			
			
		}
		
	}
	
}