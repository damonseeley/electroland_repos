package net.electroland.kioskengine  {
	import flash.net.URLRequest;
	import flash.net.URLLoader;
    import flash.net.URLVariables;
	import flash.events.Event;
	import flash.events.ProgressEvent;
	import flash.events.SecurityErrorEvent;
	import flash.events.HTTPStatusEvent;
	import flash.events.IOErrorEvent;
	
	public class Logger{
		
		private var started:Boolean = false;
		private var loggerURL:String;
		private var sessionID:String;
		
		public function Logger(){
			//loggerURL = "http://datadreamer.dyndns.org/ive/logger.php";
		}
		
		public function start(filename:String):void{
			if(!started){
				started = true;
			}
		}
		
		public function logEvent(sysTime:String, seshTime:String, kioskID:String, seshID:String,
								 lang:String, sex:String, age:String, eventType:String, modName:String,
								 unitName:String, itemName:String, itemDesc:String, eventDesc:String):void{
			// SESHID IS NO LONGER USED, USING SESSIONID INSTEAD
			var newDate:Date = new Date(Date.parse(sysTime));
			sysTime = newDate.getFullYear() +"-"+ (newDate.getMonth()+1) +"-"+ newDate.getDate() + " "+ newDate.getHours() +":"+ newDate.getMinutes() +":"+ newDate.getSeconds();
			trace(sysTime);
			var entry:String = "?sysTime="+sysTime+"&seshTime="+seshTime+"&kioskID="+kioskID+"&seshID="+sessionID+"&lang="+lang+"&sex="+sex+"&age="+age+"&eventType="+eventType+"&modName="+modName+"&unitName="+unitName+"&itemName="+itemName+"&itemDesc="+itemDesc+"&eventDesc="+eventDesc;
			var request:URLRequest = new URLRequest(loggerURL+"logger.php"+entry);
			trace(loggerURL+"logger.php"+entry);
			var loader:URLLoader = new URLLoader();
			try {
                loader.load(request);
            } catch (error:Error) {
                //trace("Unable to load requested document.");
            }
		}
		
		private function completeHandler(event:Event):void {
            var loader:URLLoader = URLLoader(event.target);
            //trace("completeHandler: " + loader.data);
        }
		
		private function newSessionID(event:Event):void {
            var loader:URLLoader = URLLoader(event.target);
            trace("newSessionID: " + loader.data);
			sessionID = loader.data;
        }

        private function openHandler(event:Event):void {
            //trace("openHandler: " + event);
        }

        private function progressHandler(event:ProgressEvent):void {
            //trace("progressHandler loaded:" + event.bytesLoaded + " total: " + event.bytesTotal);
        }

        private function securityErrorHandler(event:SecurityErrorEvent):void {
            //trace("securityErrorHandler: " + event);
        }

        private function httpStatusHandler(event:HTTPStatusEvent):void {
            //trace("httpStatusHandler: " + event);
        }

        private function ioErrorHandler(event:IOErrorEvent):void {
            //trace("ioErrorHandler: " + event);
        }
		
		public function setURL(loggerURL:String):void{
			this.loggerURL = loggerURL;
		}
		
		public function getSessionID(kioskID:Number):void{
			var request:URLRequest = new URLRequest(loggerURL+"getSessionID.php?kioskID="+kioskID+"&ck="+String(Math.random()));
			var loader:URLLoader = new URLLoader();
			loader.addEventListener("complete", newSessionID);
			try {
                loader.load(request);
            } catch (error:Error) {
                //trace("Unable to load requested document.");
            }
		}
		
		
		public function close():void{
			// TODO: remove this from conductor and here
		}

		
	}
	
	
	//import flash.filesystem.*;
//	
//	public class Logger{
//		private var filename:String;
//		private var logDir:File;
//		private var file:File;
//		private var stream:FileStream;
//		private var started:Boolean = false;
//		
//		/*
//		
//		LOGGER.as
//		by Aaron Siegel, 7-1-09
//		
//		Receives an event every time an ItemEvent is dispatched, and timestamps it
//		for a file that will be used to log activity from each kiosk session.
//		
//		*/
//		
//		public function Logger(){
//			logDir = File.desktopDirectory.resolvePath("LOGS");
//		}
//		
//		public function start(filename:String):void{
//			if(!started){
//				started = true;
//				this.filename = filename;		
//				file = logDir.resolvePath(filename);
//				open();
//			}
//		}
//		
//		/*
//		public function logEvent(event:ItemEvent):void{
//			// timestamp and log this to file
//			trace("## LOG EVENT: "+event.getAction() +" "+ event.getArguments());
//		}
//		*/
//		
//		public function logEvent(sysTime:String, seshTime:String, kioskID:String, seshID:String,
//								 lang:String, sex:String, age:String, eventType:String, modName:String,
//								 unitName:String, itemName:String, itemDesc:String, eventDesc:String):void{
//			// system timestamp, session timestamp, kiosk ID#, session ID#, language, sex, age group, event type, module name, unit name, item name, item description, event description
//			if(started){
//				write(sysTime +","+ seshTime +","+ kioskID +","+ seshID +","+ lang +","+ sex +","+ age +","+ eventType +","+ modName +","+ unitName +","+ itemName +","+ itemDesc +","+ eventDesc+"\n");
//			}
//		}
//		
//		public function close():void{
//			if(started){
//				stream.close();
//				stream = null;
//				started = false;
//			}
//		}
//		
//		public function open():void{
//			stream = new FileStream();
//			stream.open( file, FileMode.APPEND );
//			write("# system timestamp, session duration, kiosk ID, session ID, language, sex, age group, event type, module name, unit name, item name, item description, event description\n");
//		}
//		
//		public function write(msg:String):void{
//			if(stream != null){
//				stream.writeMultiByte(msg, "iso-8859-1");
//			}
//		}
//		
//	}
	
}