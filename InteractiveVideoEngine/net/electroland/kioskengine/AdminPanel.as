package net.electroland.kioskengine {
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.system.System;
	import flash.desktop.NativeApplication;
	import flash.events.*;
	
	public class AdminPanel extends Sprite {
		private var conductor:Conductor;
		private var bg:Sprite;
		private var englishButton:Sprite;
		private var spanishButton:Sprite;
		private var maleButton:Sprite;
		private var femaleButton:Sprite;
		private var youngButton:Sprite;
		private var middleButton:Sprite;
		private var oldButton:Sprite;
		private var quitButton:Sprite;
		private var exitButton:Sprite;
		private var titleLabel:TextField;
		private var textFormat:TextFormat;
		private var buttonTextFormat:TextFormat;
		
		/*
		
		ADMINPANEL.as
		by Aaron Siegel, 7-9-09
		
		Provides various administrative controls
		
		*/
		
		public function AdminPanel(conductor:Conductor){
			this.conductor = conductor;
			this.visible = false;
			bg = new Sprite();
			addChild(bg);
			bg.graphics.lineStyle(0,0xaaaaaa);
			bg.graphics.beginFill(0x000000, 0.9);
			bg.graphics.drawRect(0,0,conductor.parent.stage.stageWidth,conductor.parent.stage.stageHeight);
			bg.graphics.endFill();
			
			titleLabel = new TextField();
			titleLabel.selectable = false;
			titleLabel.blendMode = BlendMode.LAYER;
			titleLabel.autoSize = TextFieldAutoSize.LEFT;
			titleLabel.x = 10;
			titleLabel.y = 10;
			//titleLabel.embedFonts = true;
			titleLabel.text = "Settings";
			textFormat = new TextFormat();
			textFormat.font = "Arial";
			textFormat.size = 24;
			textFormat.bold = true;
			textFormat.color = 0xffffff;
			titleLabel.setTextFormat(textFormat);
			addChild(titleLabel);
			
			buttonTextFormat = new TextFormat();
			buttonTextFormat.font = "Arial";
			buttonTextFormat.size = 14;
			buttonTextFormat.bold = true;
			buttonTextFormat.color = 0xffffff;
			
			
			englishButton = newButton("English");
			addChild(englishButton);
			englishButton.addEventListener(MouseEvent.MOUSE_DOWN, setLanguageEnglish);
			englishButton.x = 15;
			englishButton.y = 75;
			
			spanishButton = newButton("Spanish");
			addChild(spanishButton);
			spanishButton.addEventListener(MouseEvent.MOUSE_DOWN, setLanguageSpanish);
			spanishButton.x = 130;
			spanishButton.y = 75;
			
			if(conductor.getLanguage() == "English"){
				spanishButton.alpha = 0.5;
			} else if(conductor.getLanguage() == "Spanish"){
				englishButton.alpha = 0.5;
			}
			
			maleButton = newButton("Male");
			addChild(maleButton);
			maleButton.addEventListener(MouseEvent.MOUSE_DOWN, setGenderMale);
			maleButton.x = 15;
			maleButton.y = 190;
			
			femaleButton = newButton("Female");
			addChild(femaleButton);
			femaleButton.addEventListener(MouseEvent.MOUSE_DOWN, setGenderFemale);
			femaleButton.x = 130;
			femaleButton.y = 190;
			
			if(conductor.getGender() == "Male"){
				femaleButton.alpha = 0.5;
			} else if(conductor.getGender() == "Female"){
				maleButton.alpha = 0.5;
			}
			
			youngButton = newButton("Young");
			addChild(youngButton);
			youngButton.addEventListener(MouseEvent.MOUSE_DOWN, setAgeYoung);
			youngButton.x = 15;
			youngButton.y = 305;
			
			middleButton = newButton("Middle");
			addChild(middleButton);
			middleButton.addEventListener(MouseEvent.MOUSE_DOWN, setAgeMiddle);
			middleButton.x = 130;
			middleButton.y = 305;
			
			oldButton = newButton("Old");
			addChild(oldButton);
			oldButton.addEventListener(MouseEvent.MOUSE_DOWN, setAgeOld);
			oldButton.x = 245;
			oldButton.y = 305;
			
			if(conductor.getAge() == "Young"){
				youngButton.alpha = 1;
				middleButton.alpha = 0.5;
				oldButton.alpha = 0.5;
			} else if(conductor.getAge() == "Middle"){
				youngButton.alpha = 0.5;
				middleButton.alpha = 1;
				oldButton.alpha = 0.5;
			} else if(conductor.getAge() == "Old"){
				youngButton.alpha = 0.5;
				middleButton.alpha = 0.5;
				oldButton.alpha = 1;
			}
			
			quitButton = newButton("Quit");
			addChild(quitButton);
			quitButton.addEventListener(MouseEvent.MOUSE_DOWN, quit);
			quitButton.x = conductor.parent.stage.stageWidth - 115;
			quitButton.y = conductor.parent.stage.stageHeight - 300;
			
			exitButton = newButton("Exit");
			addChild(exitButton);
			exitButton.addEventListener(MouseEvent.MOUSE_DOWN, exit);
			exitButton.x = conductor.parent.stage.stageWidth - 115;
			exitButton.y = conductor.parent.stage.stageHeight - 115;
			
		}
		
		public function quit(event:MouseEvent):void{
			NativeApplication.nativeApplication.exit()
		}
		
		public function exit(event:MouseEvent):void{
			toggleDisplay();
		}
		
		public function setLanguageEnglish(event:MouseEvent):void{
			conductor.setLanguage("English");
			englishButton.alpha = 1;
			spanishButton.alpha = 0.5;
		}
		
		public function setLanguageSpanish(event:MouseEvent):void{
			conductor.setLanguage("Spanish");
			spanishButton.alpha = 1;
			englishButton.alpha = 0.5;
		}
		
		public function setGenderMale(event:MouseEvent):void{
			conductor.setGender("Male");
			maleButton.alpha = 1;
			femaleButton.alpha = 0.5;
		}
		
		public function setGenderFemale(event:MouseEvent):void{
			conductor.setGender("Female");
			femaleButton.alpha = 1;
			maleButton.alpha = 0.5;
		}
		
		public function setAgeYoung(event:MouseEvent):void{
			conductor.setAge("Young");
			youngButton.alpha = 1;
			middleButton.alpha = 0.5;
			oldButton.alpha = 0.5;
		}
		
		public function setAgeMiddle(event:MouseEvent):void{
			conductor.setAge("Middle");
			youngButton.alpha = 0.5;
			middleButton.alpha = 1;
			oldButton.alpha = 0.5;
		}
		
		public function setAgeOld(event:MouseEvent):void{
			conductor.setAge("Old");
			youngButton.alpha = 0.5;
			middleButton.alpha = 0.5;
			oldButton.alpha = 1;
		}
		
		public function toggleDisplay():void{
			this.visible = !this.visible;
		}
		
		public function newButton(textValue:String):Sprite{
			var button:Sprite = new Sprite();
			var square:Sprite = new Sprite();
			var textLabel:TextField = new TextField();
			square.graphics.lineStyle(3,0xffffff);
			square.graphics.beginFill(0xaaaaaa, 0.9);
			square.graphics.drawRoundRect(0,0,100,100,20);
			square.graphics.endFill();
			button.addChild(square);
			textLabel.selectable = false;
			textLabel.blendMode = BlendMode.LAYER;
			textLabel.autoSize = TextFieldAutoSize.CENTER;
			textLabel.text = textValue;
			textLabel.setTextFormat(buttonTextFormat);
			button.addChild(textLabel);
			textLabel.x = square.width/2 - textLabel.width/2;
			textLabel.y = square.height/2 - textLabel.height/2;
			return button;
		}		
		
		
	}
	
}