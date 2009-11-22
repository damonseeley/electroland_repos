import processing.core.*; 
import processing.xml.*; 

import controlP5.*; 
import processing.opengl.*; 
import java.util.concurrent.ConcurrentHashMap; 
import mpe.client.*; 
import java.text.*; 
import java.awt.geom.Rectangle2D; 
import com.sun.opengl.util.*; 
import com.sun.opengl.util.j2d.*; 

import mpe.config.*; 
import controlP5.*; 
import mpe.client.*; 

import java.applet.*; 
import java.awt.*; 
import java.awt.image.*; 
import java.awt.event.*; 
import java.io.*; 
import java.net.*; 
import java.text.*; 
import java.util.*; 
import java.util.zip.*; 
import java.util.regex.*; 

public class FILMain extends PApplet {






// FIL TEXT CLOUD
// by Electroland
//
// Java/Processing development by Aaron Siegel, October/November 2009.
//
// Projected Area: 100'2" x 18'8" = 3072 x 768
// Interface Area: 40" diagonal   = 1920 x 1080
// Ramp Area:      54'3" x 6'5" ramped down to 2'7"



PGraphicsOpenGL pgl;             // opengl stuff
ControlP5 controlP5;             // controls for debugging
TCPClient client;                // used for MPE communication

// DATA STORAGE VARIABLES
Properties properties            = new Properties();
ConcurrentHashMap textBlocks     = new ConcurrentHashMap();
ConcurrentHashMap authorObjects  = new ConcurrentHashMap();
ConcurrentHashMap quoteObjects   = new ConcurrentHashMap();
ConcurrentHashMap bioObjects     = new ConcurrentHashMap();
ArrayList genreList_english      = new ArrayList();
ArrayList genreList_spanish      = new ArrayList();
int numAuthors                   = 0;
String xmlFileName;

// APPLET DIMENSION VARIABLES
int xSize, ySize;                // stage width/height
float xGrav, yGrav;              // location of gravity center
float xgravity, ygravity;        // gravitational force
float interfaceScale = 1.6f;      // scaling of the text cloud for master control machine
float defaultInterfaceScale;
int verticalOffset = 0;
float horizontalOffset = 0;        // necessary for slider widget to operate
boolean yflip = false;
boolean standAlone = false;      // for testing without MPE

// TEXT VARIABLES
String authorFontName, quoteFontName, bioFontName;  // font name
String dateFontName, genreFontName;
int authorFontSize, quoteFontSize, bioFontSize;     // font size
int dateFontSize, genreFontSize;
float textScaleMin, textScaleMax;                   // scaling of textblocks (0-1 being 100%)
float activatedScaleMax;                            // the scale author names inflate to when activated
float textMarginVertical, textMarginHorizontal;     // margin around text blocks
float quoteMarginHorizontal, quoteMarginVertical;   // specific buffer just for quotes/bios
int overflow;                                       // amount of text overflow on each side of stage before wrapping
int numTextBlocks, authorNum;                       // keeping track of textblock creation since application start
float textVerticalDamping, textHorizontalDamping;   // default motion damping for all text
float textHorizontalSpring, textVerticalSpring;     // multiplier for moving textblocks away from each other
int textRollOverDuration, textRollOutDuration;      // author name drag-over effect
float textRollOverScale, textRollOverAlpha;         // scale and opacity multipliers when dragged over
float quoteTextScale;                               // scale for quote text
float bioTextScale;
float dateTextScale;
float genreTextScale;
int quoteIntroDelay;                                // delay to wait for textblocks to get pushed out before revealing quote
int quoteFadeInDuration;
int quoteHoldDuration;
int quoteFadeOutDuration;
int quoteDistanceThreshold;
int quotationOffset;
int authorFadeOutDelay;    // additional delay author name waits AFTER quote/bio has disappeared from stage
int charsPerLine;
int quoteID = 0;
int maxQuotes;
float clearAreaMultiplier;
float quotePushMultiplier;
int quoteBlockTopMargin, bioBlockTopMargin, dateTextLeftMargin, genreTextTopMargin;
int quoteMinDisplay;

// COLOR VARIABLES
int backgroundGray;
int authorTextRedVal, authorTextGreenVal, authorTextBlueVal;
int authorSelectedRedVal, authorSelectedGreenVal, authorSelectedBlueVal;
int quoteTextRedVal, quoteTextGreenVal, quoteTextBlueVal;
int bioTextRedVal, bioTextGreenVal, bioTextBlueVal;
int dateTextRedVal, dateTextGreenVal, dateTextBlueVal;
int genreTextRedVal, genreTextGreenVal, genreTextBlueVal;
int textAlphaMax, textAlphaFallOff;
int horizontalFallOff;

// CONTROL VARIABLES
float dragDamp;                  // slows users dragging action
float dragRadius;
float postDragDelay;             // prevents accidental release
boolean addAuthor = false;
boolean displayStats = false;
boolean displayControls = false;
boolean displayFrames = false;
boolean displayBoundingBoxes = false;
boolean enableCamera = false;
boolean zooming = false;
boolean displayCursor = false;
int zoomCounter, zoomDuration, zoomDelayCounter;
int buttonZoomDuration, inactivityZoomDuration;
float zoomTarget, zoomStart;
float inactivityZoomDelay;
float maxZoom, minZoom;
PFont statFont;
long lastDragged;
int cameraX, cameraY, cameraZ;
WidgetManager widgetManager;
Button btnEnglish, btnEspanol, btnAuthorCloud, btnDate, btnPlus, btnMinus, btnPopularity;
DropDownMenu dropdownGenre;
Balloon balloon;
Slider slider;
float horizontalMouseOffset, verticalMouseOffset;
float scaledWidth, scaledHeight;
int barSlideDuration;  // speed of sliderbar auto-movement

// BUTTON VARIABLES
PImage buttonAuthorCloudImage, buttonDateImage, buttonGenreImage, buttonPopularityImage;
PImage buttonAuthorCloudDown, buttonDateDown, buttonGenreDown, buttonPopularityDown;
int buttonAuthorCloudX, buttonAuthorCloudY, buttonDateX, buttonDateY;
int buttonGenreX, buttonGenreY, buttonPopularityX, buttonPopularityY;
PImage buttonEnglishImage, buttonEspanolImage, buttonPlusImage, buttonMinusImage, buttonSliderImage;
PImage buttonEnglishDown, buttonEspanolDown, buttonPlusDown, buttonMinusDown;
int buttonEnglishX, buttonEnglishY, buttonEspanolX, buttonEspanolY;
int buttonPlusX, buttonPlusY, buttonMinusX, buttonMinusY;
int buttonSliderX, buttonSliderY, dropDownItemHeight, dropDownItemLeading;
PImage buttonQuoteImage, buttonQuoteDown, buttonBiographyImage, buttonBiographyDown, balloonImage, balloonDown;
PImage buttonQuoteEspImage, buttonQuoteEspDown, buttonBiographyEspImage, buttonBiographyEspDown, buttonQuoteGrey, buttonBiographyGrey;
PImage backgroundImage, leftFade, rightFade;
PImage sliderBarSegmentImage, sliderBarSegmentDown, sliderBarLeftImage, sliderBarLeftDown, sliderBarRightImage, sliderBarRightDown;

// IMAGE VARIABLES
PImage imageLanguageInstructionsImage, imageViewingInstructionsImage, imageZoomTitleImage, imageTitleImage;
int imageLanguageInstructionsX, imageLanguageInstructionsY, imageViewingInstructionsX, imageViewingInstructionsY;
int imageZoomTitleX, imageZoomTitleY, imageTitleX, imageTitleY;

// USER VARIABLES
String userLanguage = "English"; // "Espanol";
String mode = "authorcloud";     // "authorcloud", "date", "genre", "popularity"
String currentGenre;             // name of genre being displayed

// SCREENSAVER VARIABLES
int screensaverInactivity;              // number of frames until screensaver is turned on
float screensaverSpeed;                 // velocity applied to each text block
boolean screensaverActivated = false;   // boolean to kick on application of velocity
int inactivityCounter = 0;              // keep track of last user action
int resetDelay;                         // # of frames to wait after screensaver is activated to reset everything
int resetCounter = 0;                   // counter to reset
int resetDuration;

// STAT VARIABLES
long lastTime;
int mpeFps = 0;
ArrayList mpeFpsHistory = new ArrayList();

// RAMP MASK VARIABLES
RampMask rampMask;
boolean displayRamp = false;
boolean enableRampForce = false;
int rampMaskTopLeftX, rampMaskTopLeftY;
int rampMaskTopRightX, rampMaskTopRightY;
int rampMaskBottomRightX, rampMaskBottomRightY;
int rampMaskBottomLeftX, rampMaskBottomLeftY;






// INIT FUNCTIONS

public void setup(){    
  loadProperties();
  
  client = new TCPClient(sketchPath("mpe.ini"), this);
  size(client.getLWidth(), client.getLHeight(), OPENGL);
  randomSeed(1);
  if(!displayCursor){
    noCursor();
  }
  xGrav = client.getMWidth()/2;
  yGrav = client.getMHeight()/2;
  colorMode(RGB, 255);
  pgl = (PGraphicsOpenGL)g;
  statFont = loadFont("ArialMT-10.vlw");
  
  if(displayControls){
    loadControls();
  }
  
  rampMask = new RampMask(rampMaskTopLeftX, rampMaskTopLeftY, rampMaskTopRightX, rampMaskTopRightY, rampMaskBottomRightX, rampMaskBottomRightY, rampMaskBottomLeftX, rampMaskBottomLeftY);
  
  // TEMPORARY! MUST GET GENRE LIST FROM BILINGUAL FILE
  genreList_english.add("fiction");
  genreList_english.add("nonfiction");
  
  loadXML(xmlFileName);
  if(!standAlone){
    client.start();
  }
  lastTime = System.currentTimeMillis();  // record time for FPS comparison
}

public void loadXML(String filename){
  XMLElement xml = new XMLElement(this, sketchPath(filename));
  numAuthors = xml.getChildCount();
  for(int i=0; i<xml.getChildCount(); i++){
    XMLElement authorXML = xml.getChild(i);
    
    String name = "";
    String seminalwork_english = "";
    String seminalwork_spanish = "";
    String bio_english = "";
    String bio_spanish = "";
    String quote_english = "";
    String quote_spanish = "";
    String[] genres = new String[0];
    String[] geographies = new String[0];
    int born = 0;
    int died = 0;
    int workbegan = 0;
    int workended = 0;
      
    for(int n=0; n<authorXML.getChildCount(); n++){
      XMLElement entry = authorXML.getChild(n);
      //println(authorXML.getChild(n).getChildCount());
      
      if(entry.getContent() != null || entry.getChildCount() > 0){
        if("Name".equals(entry.getName())){
          name = entry.getContent();
        } else if("Born".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            born = Integer.parseInt(entry.getContent().substring(0,4));
          } else if ("present".equals(entry.getContent())){
            born = 2009;
          } else {
            born = Integer.parseInt(entry.getContent());
          }
        } else if("Died".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            died = Integer.parseInt(entry.getContent().substring(0,4));
          } else {
            died = Integer.parseInt(entry.getContent());
          }
        } else if("WorkBegan".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            workbegan = Integer.parseInt(entry.getContent().substring(0,4));
          } else {
            workbegan = Integer.parseInt(entry.getContent());
          }
        } else if("WorkEnded".equals(entry.getName())){
          if(entry.getContent().endsWith("s")){
            workended = Integer.parseInt(entry.getContent().substring(0,4));
          } else if ("present".equals(entry.getContent())){
            // work has not ended, so leave it null
          } else {
            workended = Integer.parseInt(entry.getContent());
          }
        } else if("SeminalWork".equals(entry.getName())){
          // TODO: switch to english/spanish
          //seminalwork_english = entry.getContent();
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                seminalwork_english = entry.getChild(q).getContent();
              } else if("Spanish".equals(entry.getChild(q).getName())){
                seminalwork_spanish = entry.getChild(q).getContent();
              }
            }
          }
        } else if("AuthorDescription".equals(entry.getName())){
          // TODO: switch to Biography and english/spanish
          //bio_english = entry.getContent();
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                bio_english = entry.getChild(q).getContent();
              } else if("Spanish".equals(entry.getChild(q).getName())){
                bio_spanish = entry.getChild(q).getContent();
              }
            }
          }
        } else if("Geographies".equals(entry.getName())){
          geographies = trim(entry.getContent().split(","));
        } else if("Genres".equals(entry.getName())){
          genres = trim(entry.getContent().split(","));
        } else if("Quote".equals(entry.getName())){
          for(int q=0; q<entry.getChildCount(); q++){
            if(entry.getChild(q).getContent() != null){
              if("English".equals(entry.getChild(q).getName())){
                quote_english = entry.getChild(q).getContent();
                //if(quote_english.split("\\*").length > 1){
                  //println(quote_english);
                //}
              } else if("Spanish".equals(entry.getChild(q).getName())){
                quote_spanish = entry.getChild(q).getContent();
              }
            }
          }
        }
      }
      
    }
    // CREATE AUTHOR HERE
    createAuthor(i, name, born, died, workbegan, workended, seminalwork_english, seminalwork_spanish, geographies, genres, quote_english, quote_spanish, bio_english, bio_spanish);
  }
}

public void createAuthor(int id, String name, int born, int died, int workbegan, int workended,
                  String seminalwork_english, String seminalwork_spanish, String[] geographies,
                  String[] genres, String quote_english, String quote_spanish, String bio_english,
                  String bio_spanish){
                    
  float xpos = ((id % 10) * ((client.getMWidth()+(overflow*2)) / 10)) - overflow;
  float ypos;
  if((id/10) % 2 > 0){
    ypos = (client.getMHeight()/2) - ((id / 10) * 10);
  } else {
    ypos = (client.getMHeight()/2) + ((id / 10) * 10);
  }
  // ORIGINAL SCALING ALGORITHM
  float textscale = random(textScaleMin, textScaleMin + (((100-id) * 0.01f) * (textScaleMax-textScaleMin)));
  // UNUSED NEW SCALING ALGORITHM
  //float scaleRange = (((numAuthors-id) * (1.0/numAuthors)) * (textScaleMax-textScaleMin));
  //println(scaleRange);
  //float textscale = random(textScaleMin, textScaleMin + scaleRange);
  
  Author author = new Author(id, name, quote_english, xpos, ypos, authorFontName, authorFontSize, textscale);
  author.setRed(authorTextRedVal);
  author.setGreen(authorTextGreenVal);
  author.setBlue(authorTextBlueVal);
  author.setAlphaMax(textAlphaMax);
  author.setAlphaFallOff(textAlphaFallOff);
  author.setXMargin(textMarginHorizontal);
  author.setYMargin(textMarginVertical);
  author.setHorizontalDamping(textHorizontalDamping);
  author.setVerticalDamping(textVerticalDamping);
  author.setHorizontalSpring(textHorizontalSpring);
  author.setVerticalSpring(textVerticalSpring);
  author.setRollOverDuration(textRollOverDuration);
  author.setRollOutDuration(textRollOutDuration);
  author.setRollOverScale(textRollOverScale);
  author.setRollOverAlpha(textRollOverAlpha);
  author.setSelectedRed(authorSelectedRedVal);
  author.setSelectedGreen(authorSelectedGreenVal);
  author.setSelectedBlue(authorSelectedBlueVal);
  author.setStageWidth(client.getMWidth());
  author.setStageHeight(client.getMHeight());
  author.setMaxScale(activatedScaleMax);
  author.softenCollisions();  // tweens damping
  
  author.born = born;
  author.died = died;
  author.workbegan = workbegan;
  author.workended = workended;
  author.seminalwork_english = seminalwork_english;
  author.seminalwork_spanish = seminalwork_spanish;
  author.geographies = geographies;
  author.genres = genres;
  author.quote_english = quote_english;
  author.quote_spanish = quote_spanish;
  author.bio_english = bio_english;
  author.bio_spanish = bio_spanish;
  
  textBlocks.put(id, author);
  authorObjects.put(id, author);
  numTextBlocks++;
  authorNum++;
}

public void loadControls(){  
  widgetManager   = new WidgetManager(client);
  btnEnglish      = new Button("English", buttonEnglishX, buttonEnglishY, 1, buttonEnglishImage, buttonEnglishDown);
  btnEnglish.silentOn();      // defaults to on
  btnEspanol      = new Button("Espanol", buttonEspanolX, buttonEspanolY, 0, buttonEspanolImage, buttonEspanolDown);
  //btnAuthorCloud  = new Button("Author Cloud", buttonAuthorCloudX, buttonAuthorCloudY, 2, buttonAuthorCloudImage, buttonAuthorCloudDown);
  //btnAuthorCloud.silentOn();  // defaults to on
  //btnDate         = new Button("By Date", buttonDateX, buttonDateY, 3, buttonDateImage, buttonDateDown);
  //btnPopularity   = new Button("By Popularity", buttonPopularityX, buttonPopularityY, 4, buttonPopularityImage, buttonPopularityDown);
  //dropdownGenre   = new DropDownMenu("By Genre", buttonGenreX, buttonGenreY, 5, dropDownItemHeight, dropDownItemLeading, buttonGenreImage, buttonGenreDown);
  //dropdownGenre.addItem("Fiction", 0);  // these will be added automatically from a props file
  //dropdownGenre.addItem("Non-Fiction", 1);
  btnPlus         = new Button("ZoomIn", buttonPlusX, buttonPlusY, 6, buttonPlusImage, buttonPlusDown);
  btnMinus        = new Button("ZoomOut", buttonMinusX, buttonMinusY, 7, buttonMinusImage, buttonMinusDown);
  slider          = new Slider("Slider", buttonSliderX, buttonSliderY, 8, buttonSliderImage, sliderBarSegmentImage, sliderBarSegmentDown, sliderBarLeftImage, sliderBarLeftDown, sliderBarRightImage, sliderBarRightDown);
  scaledWidth = client.getMWidth() * (1/interfaceScale);
  scaledHeight = client.getMHeight() * (1/interfaceScale);
  horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
  verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
  slider.setAreaVisible(width/(float)scaledWidth);
  slider.setOffset(horizontalMouseOffset/(float)scaledWidth);
  slider.setBarSlideDuration(barSlideDuration);
  
  widgetManager.addItem(btnEnglish);
  widgetManager.addItem(btnEspanol);
  //widgetManager.addItem(btnAuthorCloud);
  //widgetManager.addItem(btnDate);
  //widgetManager.addItem(btnPopularity);
  //widgetManager.addItem(dropdownGenre);
  widgetManager.addItem(btnPlus);
  widgetManager.addItem(btnMinus);
  widgetManager.addItem(slider);
}

public void loadProperties(){
  try{
    //InputStream in = createInput("properties.txt");  // load in the properties for this project
    InputStream in = createInput(sketchPath("properties.txt"));
    properties.load(in);
    in.close();
  } catch(Exception e){
    e.printStackTrace();
    System.exit(0);
  }

  xmlFileName             = properties.getProperty("xmlFileName");
  overflow                = Integer.parseInt(properties.getProperty("overflow"));  // amount of text overflow on each side of stage before wrapping
  xgravity                = Float.parseFloat(properties.getProperty("xgravity"));  // gravitational force on text blocks
  ygravity                = Float.parseFloat(properties.getProperty("ygravity"));
  yflip                   = Boolean.parseBoolean(properties.getProperty("yflip"));
  interfaceScale          = Float.parseFloat(properties.getProperty("interfaceScale"));
  defaultInterfaceScale   = interfaceScale;
  //verticalOffset        = Integer.parseInt(properties.getProperty("verticalOffset"));
  dragDamp                = Float.parseFloat(properties.getProperty("dragDamp"));  // amount of damping user control
  dragRadius              = Float.parseFloat(properties.getProperty("dragRadius"));
  postDragDelay           = Float.parseFloat(properties.getProperty("postDragDelay"));
  dropDownItemHeight      = Integer.parseInt(properties.getProperty("dropDownItemHeight"));
  dropDownItemLeading     = Integer.parseInt(properties.getProperty("dropDownItemLeading"));
  //standAlone              = Boolean.parseBoolean(properties.getProperty("standAlone"));
  buttonZoomDuration      = Integer.parseInt(properties.getProperty("zoomDuration"));
  maxZoom                 = Float.parseFloat(properties.getProperty("maxZoom"));
  minZoom                 = Float.parseFloat(properties.getProperty("minZoom"));
  barSlideDuration        = Integer.parseInt(properties.getProperty("barSlideDuration"));
  
  displayRamp             = Boolean.parseBoolean(properties.getProperty("displayRamp"));
  enableRampForce         = Boolean.parseBoolean(properties.getProperty("enableRampForce"));
  rampMaskTopLeftX        = Integer.parseInt(properties.getProperty("rampMaskTopLeftX"));
  rampMaskTopLeftY        = Integer.parseInt(properties.getProperty("rampMaskTopLeftY"));
  rampMaskTopRightX       = Integer.parseInt(properties.getProperty("rampMaskTopRightX"));
  rampMaskTopRightY       = Integer.parseInt(properties.getProperty("rampMaskTopRightY"));
  rampMaskBottomRightX    = Integer.parseInt(properties.getProperty("rampMaskBottomRightX"));
  rampMaskBottomRightY    = Integer.parseInt(properties.getProperty("rampMaskBottomRightY"));
  rampMaskBottomLeftX     = Integer.parseInt(properties.getProperty("rampMaskBottomLeftX"));
  rampMaskBottomLeftY     = Integer.parseInt(properties.getProperty("rampMaskBottomLeftY"));
  
  backgroundGray          = Integer.parseInt(properties.getProperty("backgroundGray"));      // color properties
  authorTextRedVal        = Integer.parseInt(properties.getProperty("authorTextRedVal"));
  authorTextGreenVal      = Integer.parseInt(properties.getProperty("authorTextGreenVal"));
  authorTextBlueVal       = Integer.parseInt(properties.getProperty("authorTextBlueVal"));
  textAlphaMax            = Integer.parseInt(properties.getProperty("textAlphaMax"));
  textAlphaFallOff        = Integer.parseInt(properties.getProperty("textAlphaFallOff"));
  horizontalFallOff       = Integer.parseInt(properties.getProperty("horizontalFallOff"));
  authorSelectedRedVal    = Integer.parseInt(properties.getProperty("authorSelectedRedVal"));
  authorSelectedGreenVal  = Integer.parseInt(properties.getProperty("authorSelectedGreenVal"));
  authorSelectedBlueVal   = Integer.parseInt(properties.getProperty("authorSelectedBlueVal"));
  quoteTextRedVal         = Integer.parseInt(properties.getProperty("quoteTextRedVal"));
  quoteTextGreenVal       = Integer.parseInt(properties.getProperty("quoteTextGreenVal"));
  quoteTextBlueVal        = Integer.parseInt(properties.getProperty("quoteTextBlueVal"));
  quoteDistanceThreshold  = Integer.parseInt(properties.getProperty("quoteDistanceThreshold"));
  quotationOffset         = Integer.parseInt(properties.getProperty("quotationOffset"));
  quoteMinDisplay         = Integer.parseInt(properties.getProperty("quoteMinDisplay"));
  
  authorFontName        = properties.getProperty("authorFontName");                        // text properties
  authorFontSize        = Integer.parseInt(properties.getProperty("authorFontSize"));
  quoteFontName         = properties.getProperty("quoteFontName");                        
  quoteFontSize         = Integer.parseInt(properties.getProperty("quoteFontSize"));
  textScaleMin          = Float.parseFloat(properties.getProperty("textScaleMin"));
  textScaleMax          = Float.parseFloat(properties.getProperty("textScaleMax"));
  activatedScaleMax     = Float.parseFloat(properties.getProperty("activatedScaleMax"));
  quoteTextScale        = Float.parseFloat(properties.getProperty("quoteTextScale"));
  textMarginVertical    = Float.parseFloat(properties.getProperty("textMarginVertical"));
  textMarginHorizontal  = Float.parseFloat(properties.getProperty("textMarginHorizontal"));
  quoteMarginHorizontal = Float.parseFloat(properties.getProperty("quoteMarginHorizontal"));
  quoteMarginVertical   = Float.parseFloat(properties.getProperty("quoteMarginVertical"));
  charsPerLine          = Integer.parseInt(properties.getProperty("charsPerLine"));
  maxQuotes             = Integer.parseInt(properties.getProperty("maxQuotes"));
  textVerticalDamping   = Float.parseFloat(properties.getProperty("textVerticalDamping"));
  textHorizontalDamping = Float.parseFloat(properties.getProperty("textHorizontalDamping")); 
  textHorizontalSpring  = Float.parseFloat(properties.getProperty("textHorizontalSpring"));// springy action between textblocks
  textVerticalSpring    = Float.parseFloat(properties.getProperty("textVerticalSpring"));
  quotePushMultiplier   = Float.parseFloat(properties.getProperty("quotePushMultiplier"));
  quoteBlockTopMargin   = Integer.parseInt(properties.getProperty("quoteBlockTopMargin"));
  
  bioFontName           = properties.getProperty("bioFontName");
  bioFontSize           = Integer.parseInt(properties.getProperty("bioFontSize"));
  bioTextScale          = Float.parseFloat(properties.getProperty("bioTextScale"));
  bioTextRedVal         = Integer.parseInt(properties.getProperty("bioTextRedVal"));
  bioTextGreenVal       = Integer.parseInt(properties.getProperty("bioTextGreenVal"));
  bioTextBlueVal        = Integer.parseInt(properties.getProperty("bioTextBlueVal"));
  bioBlockTopMargin     = Integer.parseInt(properties.getProperty("bioBlockTopMargin"));
  
  dateFontName          = properties.getProperty("dateFontName");
  dateFontSize          = Integer.parseInt(properties.getProperty("dateFontSize"));
  dateTextScale         = Float.parseFloat(properties.getProperty("dateTextScale"));
  dateTextRedVal        = Integer.parseInt(properties.getProperty("dateTextRedVal"));
  dateTextGreenVal      = Integer.parseInt(properties.getProperty("dateTextGreenVal"));
  dateTextBlueVal       = Integer.parseInt(properties.getProperty("dateTextBlueVal"));
  dateTextLeftMargin    = Integer.parseInt(properties.getProperty("dateTextLeftMargin"));  
  
  genreFontName         = properties.getProperty("genreFontName");
  genreFontSize         = Integer.parseInt(properties.getProperty("genreFontSize"));
  genreTextScale        = Float.parseFloat(properties.getProperty("genreTextScale"));
  genreTextRedVal       = Integer.parseInt(properties.getProperty("genreTextRedVal"));
  genreTextGreenVal     = Integer.parseInt(properties.getProperty("genreTextGreenVal"));
  genreTextBlueVal      = Integer.parseInt(properties.getProperty("genreTextBlueVal"));
  genreTextTopMargin    = Integer.parseInt(properties.getProperty("genreTextTopMargin"));
  
  screensaverInactivity   = Integer.parseInt(properties.getProperty("screensaverInactivity"));
  screensaverSpeed        = Float.parseFloat(properties.getProperty("screensaverSpeed"));
  resetDelay              = Integer.parseInt(properties.getProperty("resetDelay"));
  resetDuration           = Integer.parseInt(properties.getProperty("resetDuration"));
  inactivityZoomDelay     = Integer.parseInt(properties.getProperty("inactivityZoomDelay"));
  inactivityZoomDuration  = Integer.parseInt(properties.getProperty("inactivityZoomDuration"));
  
  textRollOverDuration  = Integer.parseInt(properties.getProperty("textRollOverDuration"));
  textRollOutDuration   = Integer.parseInt(properties.getProperty("textRollOutDuration"));
  textRollOverScale     = Float.parseFloat(properties.getProperty("textRollOverScale"));
  textRollOverAlpha     = Float.parseFloat(properties.getProperty("textRollOverAlpha"));
  quoteIntroDelay       = Integer.parseInt(properties.getProperty("quoteIntroDelay"));
  quoteFadeInDuration   = Integer.parseInt(properties.getProperty("quoteFadeInDuration"));
  quoteHoldDuration     = Integer.parseInt(properties.getProperty("quoteHoldDuration"));  // duration PER line
  quoteFadeOutDuration  = Integer.parseInt(properties.getProperty("quoteFadeOutDuration"));
  authorFadeOutDelay    = Integer.parseInt(properties.getProperty("authorFadeOutDelay"));
  clearAreaMultiplier   = Float.parseFloat(properties.getProperty("clearAreaMultiplier"));
  
  enableCamera          = Boolean.parseBoolean(properties.getProperty("enableCamera"));   // camera properties
  displayControls       = Boolean.parseBoolean(properties.getProperty("displayControls"));
  displayCursor         = Boolean.parseBoolean(properties.getProperty("displayCursor"));
  
  /*
  buttonAuthorCloudImage  = loadImage(properties.getProperty("buttonAuthorCloudImage"));                // button properties
  buttonAuthorCloudDown   = loadImage(properties.getProperty("buttonAuthorCloudDown"));
  buttonAuthorCloudX      = Integer.parseInt(properties.getProperty("buttonAuthorCloudX"));
  buttonAuthorCloudY      = Integer.parseInt(properties.getProperty("buttonAuthorCloudY"));
  buttonDateImage         = loadImage(properties.getProperty("buttonDateImage"));
  buttonDateDown          = loadImage(properties.getProperty("buttonDateDown"));
  buttonDateX             = Integer.parseInt(properties.getProperty("buttonDateX"));
  buttonDateY             = Integer.parseInt(properties.getProperty("buttonDateY"));
  buttonGenreImage        = loadImage(properties.getProperty("buttonGenreImage"));
  buttonGenreDown         = loadImage(properties.getProperty("buttonGenreDown"));
  buttonGenreX            = Integer.parseInt(properties.getProperty("buttonGenreX"));
  buttonGenreY            = Integer.parseInt(properties.getProperty("buttonGenreY"));
  buttonPopularityImage   = loadImage(properties.getProperty("buttonPopularityImage"));
  buttonPopularityDown    = loadImage(properties.getProperty("buttonPopularityDown"));
  buttonPopularityX       = Integer.parseInt(properties.getProperty("buttonPopularityX"));
  buttonPopularityY       = Integer.parseInt(properties.getProperty("buttonPopularityY"));
  */
  buttonEnglishImage      = loadImage(properties.getProperty("buttonEnglishImage"));
  buttonEnglishDown       = loadImage(properties.getProperty("buttonEnglishDown"));
  buttonEnglishX          = Integer.parseInt(properties.getProperty("buttonEnglishX"));
  buttonEnglishY          = Integer.parseInt(properties.getProperty("buttonEnglishY"));
  buttonEspanolImage      = loadImage(properties.getProperty("buttonEspanolImage"));
  buttonEspanolDown       = loadImage(properties.getProperty("buttonEspanolDown"));
  buttonEspanolX          = Integer.parseInt(properties.getProperty("buttonEspanolX"));
  buttonEspanolY          = Integer.parseInt(properties.getProperty("buttonEspanolY"));
  buttonPlusImage         = loadImage(properties.getProperty("buttonPlusImage"));
  buttonPlusDown          = loadImage(properties.getProperty("buttonPlusDown"));
  buttonPlusX             = Integer.parseInt(properties.getProperty("buttonPlusX"));
  buttonPlusY             = Integer.parseInt(properties.getProperty("buttonPlusY"));
  buttonMinusImage        = loadImage(properties.getProperty("buttonMinusImage"));
  buttonMinusDown         = loadImage(properties.getProperty("buttonMinusDown"));
  buttonMinusX            = Integer.parseInt(properties.getProperty("buttonMinusX"));
  buttonMinusY            = Integer.parseInt(properties.getProperty("buttonMinusY"));
  buttonSliderImage       = loadImage(properties.getProperty("buttonSliderImage"));
  buttonSliderX           = Integer.parseInt(properties.getProperty("buttonSliderX"));
  buttonSliderY           = Integer.parseInt(properties.getProperty("buttonSliderY"));
  buttonQuoteImage        = loadImage(properties.getProperty("buttonQuoteImage"));
  buttonQuoteDown         = loadImage(properties.getProperty("buttonQuoteDown"));
  buttonQuoteEspImage     = loadImage(properties.getProperty("buttonQuoteEspImage"));
  buttonQuoteEspDown      = loadImage(properties.getProperty("buttonQuoteEspDown"));
  buttonQuoteGrey         = loadImage(properties.getProperty("buttonQuoteGrey"));
  buttonBiographyImage    = loadImage(properties.getProperty("buttonBiographyImage"));
  buttonBiographyDown     = loadImage(properties.getProperty("buttonBiographyDown"));
  buttonBiographyEspImage = loadImage(properties.getProperty("buttonBiographyEspImage"));
  buttonBiographyEspDown  = loadImage(properties.getProperty("buttonBiographyEspDown"));
  buttonBiographyGrey     = loadImage(properties.getProperty("buttonBiographyGrey"));
  balloonImage            = loadImage(properties.getProperty("balloonImage"));
  balloonDown             = loadImage(properties.getProperty("balloonDown"));
  sliderBarSegmentImage   = loadImage(properties.getProperty("sliderBarSegmentImage"));
  sliderBarSegmentDown    = loadImage(properties.getProperty("sliderBarSegmentDown"));
  sliderBarLeftImage      = loadImage(properties.getProperty("sliderBarLeftImage"));
  sliderBarLeftDown       = loadImage(properties.getProperty("sliderBarLeftDown"));
  sliderBarRightImage     = loadImage(properties.getProperty("sliderBarRightImage"));
  sliderBarRightDown      = loadImage(properties.getProperty("sliderBarRightDown"));
  
  /*
  imageLanguageInstructionsImage  = loadImage(properties.getProperty("imageLanguageInstructionsImage"));                  // image properties
  imageLanguageInstructionsX      = Integer.parseInt(properties.getProperty("imageLanguageInstructionsX"));
  imageLanguageInstructionsY      = Integer.parseInt(properties.getProperty("imageLanguageInstructionsY"));
  imageViewingInstructionsImage   = loadImage(properties.getProperty("imageViewingInstructionsImage"));
  imageViewingInstructionsX       = Integer.parseInt(properties.getProperty("imageViewingInstructionsX"));
  imageViewingInstructionsY       = Integer.parseInt(properties.getProperty("imageViewingInstructionsY"));
  imageZoomTitleImage             = loadImage(properties.getProperty("imageZoomTitleImage"));
  imageZoomTitleX                 = Integer.parseInt(properties.getProperty("imageZoomTitleX"));
  imageZoomTitleY                 = Integer.parseInt(properties.getProperty("imageZoomTitleY"));
  imageTitleImage                 = loadImage(properties.getProperty("imageTitleImage"));
  imageTitleX                     = Integer.parseInt(properties.getProperty("imageTitleX"));
  imageTitleY                     = Integer.parseInt(properties.getProperty("imageTitleY"));
  */
  backgroundImage                 = loadImage(properties.getProperty("backgroundImage"));
  leftFade                        = loadImage(properties.getProperty("leftFade"));
  rightFade                       = loadImage(properties.getProperty("rightFade"));
}

public void createQuote(Author author){
  ArrayList quoteLines = new ArrayList();
  int wordCount = 0;
  int lineCount = 0;
  
  String quote = "";
  if(userLanguage == "English"){
    quote = trim(author.quote_english);
  } else {
    quote = trim(author.quote_spanish);
  }
  if(quote.startsWith("\"")){
    quote = quote.substring(1);
  }
  if(quote.endsWith("\"")){
    quote = quote.substring(0, quote.length()-1);
  }
  //quote = "\"" + quote + "\"";
  quote = "\u201c" + quote + "\u201d";
  
  float lineHeight = author.getTextRenderer().getHeight(quote) * quoteTextScale;
  float maxLineWidth = 0;
  
  if(quote.split("\\*").length > 1){  // PREFORMATED
    String[] lines = quote.split("\\*");
    for(int i=0; i<lines.length; i++){
      trim(lines[i]);
      float lineWidth = author.getTextRenderer().getWidth(lines[i]) * quoteTextScale;  // MUST USE QUOTE TEXT RENDERER
      //println("line width: "+lineWidth);
      if(lineWidth > maxLineWidth){
        maxLineWidth = lineWidth;
      }
      float xpos = (author.getX() - (author.getMaxWidth()*0.5f)) + (lineWidth*0.5f);  // left aligned
      float ypos = author.getY();
      float appliedQuotationOffset = 0;
      if(i == 0){
        appliedQuotationOffset = quotationOffset;
      }
      
      QuoteLine quoteLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, lines[i], xpos, ypos, quoteFontName, quoteFontSize, quoteTextScale, appliedQuotationOffset);
      quoteLine.setRed(quoteTextRedVal);
      quoteLine.setGreen(quoteTextGreenVal);
      quoteLine.setBlue(quoteTextBlueVal);
      quoteLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
      quoteLine.setFadeInDuration(quoteFadeInDuration);
      quoteLine.setFadeOutDuration(quoteFadeOutDuration);
      quoteLine.setStageWidth(client.getMWidth());
      quoteLine.setStageHeight(client.getMHeight());
      quoteLine.setPushMultiplier(quotePushMultiplier);
      quoteLine.clearArea(textBlocks, clearAreaMultiplier);
      
      textBlocks.put(numTextBlocks, quoteLine);
      quoteObjects.put(numTextBlocks, quoteLine);
      quoteLines.add(quoteLine);
      numTextBlocks++;
      lineCount++;
    }
  } else {                            // AUTOFORMATED
    String[] words = quote.split(" ");
    while(wordCount < words.length){
      int charCount = 0;
      String textLine = "";
      while(charCount < charsPerLine){
        if(wordCount < words.length){
          charCount += words[wordCount].length() + 1;
          textLine += words[wordCount] + " ";
          wordCount++;
        } else {
          break;
        }
      }
      trim(textLine);  // remove excess white space from back of line
      float lineWidth = author.getTextRenderer().getWidth(textLine) * quoteTextScale;  // MUST USE QUOTE TEXT RENDERER
      //println("line width: "+lineWidth);
      if(lineWidth > maxLineWidth){
        maxLineWidth = lineWidth;
      }
      //float xpos = random(author.getX() - (lineWidth*0.5), author.getX());  // random positioning
      //float xpos = (author.getX() - (author.getWidth()*0.5)) + (lineWidth*0.5);  // left aligned
      float xpos = (author.getX() - (author.getMaxWidth()*0.5f)) + (lineWidth*0.5f);  // left aligned
      float ypos = author.getY();
      
      float appliedQuotationOffset = 0;
      if(lineCount == 0){
        appliedQuotationOffset = quotationOffset;
      }
      
      QuoteLine quoteLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, textLine, xpos, ypos, quoteFontName, quoteFontSize, quoteTextScale, appliedQuotationOffset);
      quoteLine.setRed(quoteTextRedVal);
      quoteLine.setGreen(quoteTextGreenVal);
      quoteLine.setBlue(quoteTextBlueVal);
      quoteLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
      quoteLine.setFadeInDuration(quoteFadeInDuration);
      quoteLine.setFadeOutDuration(quoteFadeOutDuration);
      quoteLine.setStageWidth(client.getMWidth());
      quoteLine.setStageHeight(client.getMHeight());
      quoteLine.setPushMultiplier(quotePushMultiplier);
      quoteLine.clearArea(textBlocks, clearAreaMultiplier);
      
      textBlocks.put(numTextBlocks, quoteLine);
      quoteObjects.put(numTextBlocks, quoteLine);
      quoteLines.add(quoteLine);
      numTextBlocks++;
      lineCount++;
    }
  }
  
  author.setFadeInDuration(quoteFadeInDuration);  
  author.setFadeOutDuration(quoteFadeOutDuration);
  author.setAuthorFadeOutDelay(authorFadeOutDelay);
  if((quoteHoldDuration*lineCount) > quoteMinDisplay){
    author.setHoldDuration((quoteHoldDuration*lineCount) + quoteIntroDelay + authorFadeOutDelay);
  } else {
    author.setHoldDuration(quoteMinDisplay + quoteIntroDelay + authorFadeOutDelay);
  }

  ArrayList textBlocksToRemove = new ArrayList();  // list of ID numbers for textblocks to remove 
 
  // MUST FIND CENTER LOCATION OF THE ENTIRE QUOTE BLOCK
  float leftmost, rightmost, topmost, bottommost;  // top left and bottom right corners
  float centerX, centerY;                          // center of text block
  if(author.getY() < client.getMHeight()*0.5f){
    if(author.getHeight() > author.getMaxHeight()){
      leftmost = author.getX() - (author.getWidth()*0.5f);    // left side of author name and quote block
      topmost = author.getY() + (author.getHeight()*0.5f);    // bottom edge of author text, top edge of quote block
    } else {
      leftmost = author.getX() - (author.getMaxWidth()*0.5f); // left side of author name and quote block
      topmost = author.getY() + (author.getMaxHeight()*0.5f); // bottom edge of author text, top edge of quote block
    }
    rightmost = leftmost + maxLineWidth;                   // farthest right of left aligned location 
    bottommost = topmost + (lineHeight * lineCount);       // farthest down from bottom of author name
  } else {
    if(author.getHeight() > author.getMaxHeight()){
      leftmost = author.getX() - (author.getWidth()*0.5f);    // left side of author name and quote block
      bottommost = author.getY() - (author.getHeight()*0.5f); // top edge of author text, bottom edge of quote block
    } else {
      leftmost = author.getX() - (author.getMaxWidth()*0.5f);    // left side of author name and quote block
      bottommost = author.getY() + (author.getMaxHeight()*0.5f); // top edge of author text, bottom edge of quote block
    }
    rightmost = leftmost + maxLineWidth;                  // farthest right of left aligned location
    topmost = bottommost - (lineHeight * lineCount);      // top edge of top quote line
  }
  
  centerX = leftmost + ((rightmost - leftmost)/2);
  centerY = topmost + ((bottommost - topmost)/2);
  
  for(int i=0; i<quoteLines.size(); i++){
    QuoteLine quoteLine = ((QuoteLine)quoteLines.get(i));
    //Author quoteAuthor = quoteLine.getAuthor();
    //quoteLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*i)));  // this fades all lines out at once
    if((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*(lineCount-i))) > quoteMinDisplay){
      quoteLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*(lineCount-i))));    // this fades lines out as they came in
    } else {
      quoteLine.setHoldDuration(quoteMinDisplay);
    }
    //println("centerX: "+centerX +" centerY: "+ centerY);
    //println("left side: "+leftmost+" right side: "+rightmost);
    quoteLine.setParagraphCenter(centerX, centerY);
    quoteLine.setParagraphDimensions(rightmost - leftmost, bottommost - topmost);
    
    float ypos;
    if(author.getY() < client.getMHeight()*0.5f){
      //ypos = (lineHeight * i) + ((lineHeight*0.5) + author.getY()+(author.getHeight()*0.5) + textMarginVertical+1);
      if(author.getHeight() > author.getMaxHeight()){
        ypos = (lineHeight * i) + ((lineHeight*0.5f) + author.getY()+(author.getHeight()*0.5f) + textMarginVertical+1);
        ypos += quoteBlockTopMargin;
      } else {
        ypos = (lineHeight * i) + ((lineHeight*0.5f) + author.getY()+(author.getMaxHeight()*0.5f) + textMarginVertical+1);
        ypos += quoteBlockTopMargin;
      }
    } else {
      //ypos = (author.getY()-(author.getHeight()*0.5)) - ((lineCount*lineHeight) - (lineHeight*0.5) - (lineHeight * i) + (textMarginVertical+1));
      if(author.getHeight() > author.getMaxHeight()){
        ypos = (author.getY() - (author.getHeight()*0.5f)) - ((lineCount*lineHeight) - (lineHeight*0.5f) - (lineHeight * i) + (textMarginVertical+1));
        ypos -= quoteBlockTopMargin;
      } else {
        ypos = (author.getY() - (author.getMaxHeight()*0.5f)) - ((lineCount*lineHeight) - (lineHeight*0.5f) - (lineHeight * i) + (textMarginVertical+1));
        ypos -= quoteBlockTopMargin;
      }
    }
    //println(ypos);
    quoteLine.setY(ypos);
    
    // CHECK SURROUNDING AREA FOR ANY OTHER QUOTES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
    Iterator iter = quoteObjects.values().iterator();
    while(iter.hasNext()){
      QuoteLine otherQuoteLine = (QuoteLine)iter.next();
      if(otherQuoteLine.getQuoteID() != quoteLine.getQuoteID()){
        float xdist = 0;
        float ydist = 0;
        if(otherQuoteLine.getX() > quoteLine.getX()){
          xdist = (otherQuoteLine.getX() - (otherQuoteLine.getWidth()/2)) - (quoteLine.getX() + (quoteLine.getWidth()/2));
        } else {
          xdist = (quoteLine.getX() - (quoteLine.getWidth()/2)) - (otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2));
        }
        if(xdist < 0){
          xdist = 0;
        }
        
        if(otherQuoteLine.getY() > quoteLine.getY()){
          ydist = (otherQuoteLine.getY() - (otherQuoteLine.getHeight()/2)) - (quoteLine.getY() + (quoteLine.getHeight()/2));
        } else {
          ydist = (quoteLine.getY() - (quoteLine.getHeight()/2)) - (otherQuoteLine.getY() + (otherQuoteLine.getHeight()/2));
        }
        if(ydist < 0){
          ydist = 0;
        }
        
        float hypo = sqrt(sq(xdist) + sq(ydist));
        //println("hypo: "+hypo +" xdist: "+ xdist +" ydist: "+ ydist + " threshold: "+ quoteDistanceThreshold);
        if(hypo < quoteDistanceThreshold){
          textBlocksToRemove.add(otherQuoteLine.getQuoteID());
        }
      }
    }
    
    // CHECK SURROUNDING AREA FOR ANY OTHER QUOTES/BIOGRAPHIES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
    iter = bioObjects.values().iterator();
    while(iter.hasNext()){
      QuoteLine otherQuoteLine = (QuoteLine)iter.next();
      if(otherQuoteLine.getQuoteID() != quoteLine.getQuoteID()){
        float xdist = 0;
        float ydist = 0;
        if(otherQuoteLine.getX() > quoteLine.getX()){
          xdist = (otherQuoteLine.getX() - (otherQuoteLine.getWidth()/2)) - (quoteLine.getX() + (quoteLine.getWidth()/2));
        } else {
          xdist = (quoteLine.getX() - (quoteLine.getWidth()/2)) - (otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2));
        }
        if(xdist < 0){
          xdist = 0;
        }
        
        if(otherQuoteLine.getY() > quoteLine.getY()){
          ydist = (otherQuoteLine.getY() - (otherQuoteLine.getHeight()/2)) - (quoteLine.getY() + (quoteLine.getHeight()/2));
        } else {
          ydist = (quoteLine.getY() - (quoteLine.getHeight()/2)) - (otherQuoteLine.getY() + (otherQuoteLine.getHeight()/2));
        }
        if(ydist < 0){
          ydist = 0;
        }
        
        float hypo = sqrt(sq(xdist) + sq(ydist));
        //println("hypo: "+hypo +" xdist: "+ xdist +" ydist: "+ ydist + " threshold: "+ quoteDistanceThreshold);
        if(hypo < quoteDistanceThreshold){
          textBlocksToRemove.add(otherQuoteLine.getQuoteID());
        }
      }
    }
    
  }
  
  for(int i=0; i<textBlocksToRemove.size(); i++){
    int authorID = (Integer)textBlocksToRemove.get(i);
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      if(textBlock instanceof QuoteLine){
        if(((QuoteLine)textBlock).getQuoteID() == authorID){
          textBlock.fadeOutAndRemove();
        }
      } else if(textBlock instanceof Author){
        if(textBlock.getID() == authorID){
          textBlock.fadeOutAndRemove(authorFadeOutDelay);
        }
      }
    }
  }
  
  
  
  // check each one for the oldest block of text
  int oldest = Integer.MAX_VALUE;  // check each ID for lowest value
  int oldestID = 0;            // author/quote/bio shared ID
  String oldestType = "";
  ConcurrentHashMap combinedBlocks = new ConcurrentHashMap();
  Iterator iter = quoteObjects.values().iterator();  // check all quotes first
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    combinedBlocks.put(((QuoteLine)textBlock).getQuoteID(), true);  // store quote ID as key
    if(textBlock.getID() < oldest){
      oldest = textBlock.getID();
      oldestID = ((QuoteLine)textBlock).getQuoteID();
      oldestType = "Quote";
    }
  }
  iter = bioObjects.values().iterator();  // check bios
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    combinedBlocks.put(((QuoteLine)textBlock).getQuoteID(), true);  // store quote ID as key
    if(textBlock.getID() < oldest){
      oldest = textBlock.getID();
      oldestID = ((QuoteLine)textBlock).getQuoteID();
      oldestType = "Bio";
    }
  }
  
  if(combinedBlocks.size() > maxQuotes){
    if(oldestType.equals("Bio")){
      iter = bioObjects.values().iterator();  // check bios
      while(iter.hasNext()){
        QuoteLine textBlock = (QuoteLine)iter.next();
        if(textBlock.getQuoteID() == oldestID){
          textBlock.fadeOutAndRemove();
        }
      }
    } else if(oldestType.equals("Quote")){
      iter = quoteObjects.values().iterator();  // check bios
      while(iter.hasNext()){
        QuoteLine textBlock = (QuoteLine)iter.next();
        if(textBlock.getQuoteID() == oldestID){
          textBlock.fadeOutAndRemove();
        }
      }
    }
    ((Author)authorObjects.get(oldestID)).fadeOutAndRemove(authorFadeOutDelay);  // deactivate the author
  }
  
}

public void createBio(Author author){
  ArrayList bioLines = new ArrayList();
  int wordCount = 0;
  int lineCount = 0;
  String bio = "";
  if(userLanguage == "English"){
    bio = trim(author.bio_english);
  } else {
    bio = trim(author.bio_spanish);
  }
  float lineHeight = author.getTextRenderer().getHeight(bio) * quoteTextScale;  // TODO: this will not be precise if not using the quote text renderer
  String[] words = bio.split(" ");
  
  // create the genre text object that will go under the author name
  String genreString = join(author.genres, ", ").toUpperCase();
  float genreX = author.getX() + (author.getWidth()/2);
  float genreY = author.getY();
  QuoteLine genreObj = new QuoteLine(numTextBlocks, author.getID(), 0, author, genreString, genreX, genreY, genreFontName, genreFontSize, genreTextScale, 0);
  float genreHeight = genreObj.getTextRenderer().getHeight(genreString) * genreTextScale;
  /*
  // THIS SNAPS GENRE TO TOP OR BOTTOM OF AUTHOR NAME
  if(author.getY() < client.getMHeight()*0.5){
    genreY = genreY + (author.getHeight()/2) + genreHeight/2;
  } else {
    genreY = genreY - (author.getHeight()/2) - genreHeight/2;
  }
  genreObj.setY(genreY);
  */
  genreObj.setRed(genreTextRedVal);
  genreObj.setGreen(genreTextGreenVal);
  genreObj.setBlue(genreTextBlueVal);
  genreObj.setIntroDelay(0);
  genreObj.setFadeInDuration(quoteFadeInDuration);
  genreObj.setFadeOutDuration(quoteFadeOutDuration);
  genreObj.setStageWidth(client.getMWidth());
  genreObj.setStageHeight(client.getMHeight());
  genreObj.clearArea(textBlocks, clearAreaMultiplier);
  
  textBlocks.put(numTextBlocks, genreObj);
  bioObjects.put(numTextBlocks, genreObj);
  numTextBlocks++;
  
  // create the date text object that will go on the right of the author name
  String dateString = "b.";
  if(author.born > 0){
    dateString += author.born;
  }
  if(author.died > 0){
    dateString += "-" + author.died;
  }
  float dateX = author.getX();
  if(author.getScale() < activatedScaleMax){
    dateX += (author.getMaxWidth()/2);
  } else {
    dateX += (author.getWidth()/2);
  }
  float dateY = author.getY();
  QuoteLine dateObj = new QuoteLine(numTextBlocks, author.getID(), 0, author, dateString, dateX, dateY, dateFontName, dateFontSize, dateTextScale, 0);
  float dateHeight = dateObj.getTextRenderer().getHeight(dateString) * dateTextScale; 
  if(author.getY() < client.getMHeight()*0.5f){
    dateY = dateY + (author.getHeight()/2) - dateHeight/2;
  } else {
    dateY = dateY - (author.getHeight()/2) + dateHeight/2;
  }
  dateX += (dateObj.getWidth()/2) + dateTextLeftMargin;
  dateObj.setX(dateX);
  dateObj.setY(dateY);
  dateObj.setRed(dateTextRedVal);
  dateObj.setGreen(dateTextGreenVal);
  dateObj.setBlue(dateTextBlueVal);
  dateObj.setIntroDelay(0);
  dateObj.setFadeInDuration(quoteFadeInDuration);
  dateObj.setFadeOutDuration(quoteFadeOutDuration);
  dateObj.setStageWidth(client.getMWidth());
  dateObj.setStageHeight(client.getMHeight());
  dateObj.clearArea(textBlocks, clearAreaMultiplier);
  dateObj.snapToRight(dateTextLeftMargin);
  
  if(author.born > 0){
    textBlocks.put(numTextBlocks, dateObj);
    bioObjects.put(numTextBlocks, dateObj);
    numTextBlocks++;
  }
  
    
  // process the bio paragraph into separate lines
  while(wordCount < words.length){
    int charCount = 0;
    String textLine = "";
    while(charCount < charsPerLine){
      if(wordCount < words.length){
        charCount += words[wordCount].length() + 1;
        textLine += words[wordCount] + " ";
        wordCount++;
      } else {
        break;
      }
    }
    trim(textLine);  // remove excess white space from back of line
    float lineWidth = author.getTextRenderer().getWidth(textLine) * quoteTextScale;
    float xpos = (author.getX() - (author.getWidth()*0.5f)) + (lineWidth*0.5f);  // left aligned
    float ypos = author.getY();
    
    QuoteLine bioLine = new QuoteLine(numTextBlocks, author.getID(), lineCount, author, textLine, xpos, ypos, bioFontName, bioFontSize, bioTextScale, 0);
    bioLine.setRed(bioTextRedVal);
    bioLine.setGreen(bioTextGreenVal);
    bioLine.setBlue(bioTextBlueVal);
    bioLine.setIntroDelay(quoteIntroDelay + (quoteFadeInDuration*lineCount));
    bioLine.setFadeInDuration(quoteFadeInDuration);
    bioLine.setFadeOutDuration(quoteFadeOutDuration);
    bioLine.setStageWidth(client.getMWidth());
    bioLine.setStageHeight(client.getMHeight());
    bioLine.clearArea(textBlocks, clearAreaMultiplier);
    
    textBlocks.put(numTextBlocks, bioLine);
    bioObjects.put(numTextBlocks, bioLine);
    bioLines.add(bioLine);
    numTextBlocks++;
    lineCount++;
  }
  author.setFadeInDuration(quoteFadeInDuration);
  author.setFadeOutDuration(quoteFadeOutDuration);
  author.setHoldDuration((quoteHoldDuration*lineCount) + quoteIntroDelay + authorFadeOutDelay);
  author.setAuthorFadeOutDelay(authorFadeOutDelay);
  genreObj.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*lineCount)));
  dateObj.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*lineCount)));
  
  if(author.getY() < client.getMHeight()*0.5f){
    genreY = genreY + (author.getHeight()/2) + genreHeight/2;
    genreY += genreTextTopMargin;
  } else {
    genreY = genreY - (author.getHeight()/2) - genreHeight/2 - (lineCount * lineHeight);
    genreY -= genreTextTopMargin;
  }
  genreObj.setY(genreY);
  
  for(int i=0; i<bioLines.size(); i++){
    QuoteLine bioLine = ((QuoteLine)bioLines.get(i));
    //bioLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*i)));
    bioLine.setHoldDuration((quoteHoldDuration*lineCount) - (quoteIntroDelay + (quoteFadeInDuration*(lineCount-i))));    // this fades lines out as they came in
    float ypos;
    if(author.getY() < client.getMHeight()*0.5f){
      ypos = (lineHeight * i) + ((lineHeight*0.5f) + author.getY()+(author.getHeight()*0.5f) + textMarginVertical+1);
      ypos += genreHeight;
      ypos += bioBlockTopMargin;
    } else {
      ypos = (author.getY()-(author.getHeight()*0.5f)) - ((lineCount*lineHeight) - (lineHeight*0.5f) - (lineHeight * i) + (textMarginVertical+1));
      //ypos -= genreHeight;
      ypos -= bioBlockTopMargin;      
    }
    bioLine.setY(ypos);
  }
  
  // TODO: CHECK SURROUNDING AREA FOR ANY OTHER QUOTES/BIOGRAPHIES WITHIN THE THRESHOLD THAT SHOULD BE REMOVED
  
}






// FUNCTIONS FOR CHANGING VISUAL MODES

public void authorCloudMode(){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    author.moveTo(random(0, client.getMWidth()), random(0, client.getMHeight()), 60);
  }
}

public void sortByGenre(String genre){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    Boolean genrematch = false;
    for(int i=0; i<author.genres.length; i++){
      if(author.genres[i].equals(genre)){
        // move author name on to the screen
        float xpos = random(0, client.getMWidth());
        float ypos = random(0, 768);
        //println(author.getText() +" "+ genre +" "+ xpos);
        author.moveTo(xpos, ypos, 60);  // TODO: switch to props file duration
        genrematch = true;
        break;
      }
    }
    if(!genrematch){
      // send author name off the screen
      if(random(0,1) > 0.5f){
        author.moveTo(random(0, client.getMWidth()), random(client.getMHeight(), client.getMHeight()+1000), 60);
      } else {
        author.moveTo(random(0, client.getMWidth()), random(-1000, 0), 60);
      }
    }
  }
}

public void sortByDate(){
  int highest = 0;
  int lowest = 10000;
  Iterator iter = authorObjects.values().iterator();
  // loop through all authors and find the range of dates
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    // SHOULD DEFINE WHAT PROPERTY TO DO THE ANALYSIS ON
    if(author.workbegan > highest){
      highest = author.workbegan;
    } else if(author.workbegan < lowest && author.workbegan > 0){
      lowest = author.workbegan;
    }
  }
  float range = highest - lowest;
  // loop through all authors AGAIN and place them on the stage according to their date
  iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    float xpos = ((author.workbegan - lowest)/range) * client.getMWidth();
    author.moveTo(xpos, random(0,768), 60);
    //println(author.getText() +" "+ author.workbegan +" "+ xpos);
  }
}

public void sortByPopularity(){
  Iterator iter = authorObjects.values().iterator();
  while(iter.hasNext()){
    Author author = (Author)iter.next();
    float xpos = (author.popularity * 0.01f) * client.getMWidth();
    author.moveTo(xpos, random(0,768), 60);
    //println(author.getText() +" "+ author.popularity +" "+ xpos);
  }
}

public void resetCloud(){
  // (1) change all author positions and damp to their new target
  // (2) change all author text scales and tween to their new size
  Iterator iter = authorObjects.values().iterator();
  int authorCounter = 0;
  while(iter.hasNext()){
    float textscale = random(textScaleMin, textScaleMin + (((100-authorCounter) * 0.01f) * (textScaleMax-textScaleMin)));
    Author author = (Author)iter.next();
    //author.moveTo(random(0-overflow,client.getMWidth()+overflow), random(0,client.getMHeight()), resetDuration);
    author.moveTo(random(0-overflow,client.getMWidth()+overflow), random(312,768), resetDuration);
    author.scaleTo(textscale, resetDuration);
    authorCounter++;
  }
}





// RENDER THREAD

public void render(TCPClient c){
  background(backgroundGray);
  if(displayControls){
    image(backgroundImage, 0, 0);
  }
  
  // calculate zooming of interfaceScale
  if(zooming){
    float progress = sin((zoomCounter / (float)zoomDuration) * (PI/2));
    interfaceScale = zoomStart + ((zoomTarget-zoomStart) * progress);
    if(interfaceScale > minZoom){
      interfaceScale = minZoom;
    } else if(interfaceScale < maxZoom){
      interfaceScale = maxZoom;
    }
    
    //float offset = 0 - ((horizontalMouseOffset/(float)scaledWidth));
    //horizontalOffset = offset * scaledWidth;
    //println("horizontalOffset: "+ horizontalOffset +" horizontalMouseOffset: "+ horizontalMouseOffset +" offset: "+ offset);
    scaledWidth = client.getMWidth() * (1/interfaceScale);
    scaledHeight = client.getMHeight() * (1/interfaceScale);
    horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
    verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
    //println(offset * scaledWidth);
    slider.setAreaVisible(width/(float)scaledWidth);
    slider.setOffset(horizontalMouseOffset/(float)scaledWidth);
    // check slider to make sure we aren't zooming out beyond the allowed viewable area
    float offset = 0 - (slider.getBarPosition() - 0.5f);  // cloud is centered
    horizontalOffset = offset * scaledWidth;
    if(balloon != null){
      balloon.setInterfaceScale(interfaceScale);
      balloon.setHorizontalOffset(horizontalMouseOffset);
      balloon.setVerticalOffset(verticalMouseOffset);
    }
    zoomCounter++;
    //println("zoomCounter: "+ zoomCounter +" zoomDuration: "+ zoomDuration);
    if(zoomCounter >= zoomDuration){
      interfaceScale = zoomTarget;
      zooming = false;
      zoomCounter = 0;
      //println("zooming deactivated");
      if(displayControls){
        btnPlus.silentOff();
        btnMinus.silentOff();
      }
    }
  }
  
  pushMatrix();
  
  // THIS IS WHERE ALL THE SCALING AND TRANSLATING OF THE INTERFACE OCCURS, AND SHOULD NOT AFFECT PROJECTED VERSION
  if(enableCamera){
    //scale(1/interfaceScale);      // THIS IS THE STATIC METHOD (ie: no zooming)
    //translate(0,verticalOffset,0);
    translate(width/2 + horizontalOffset, height/2 + verticalOffset, 0);    // translates the center
    pushMatrix();
    scale(1/interfaceScale);                                                // scales the content to fit the screen
    translate(0-(client.getMWidth()/2), 0-(client.getMHeight()/2), 0);      // translates content to center over 0,0
    // THESE CALCULATIONS NEED TO STAY FOR WHEN THE TWEENED ZOOM EVENTS OCCUR
    scaledWidth = client.getMWidth() * (1/interfaceScale);
    scaledHeight = client.getMHeight() * (1/interfaceScale);
    horizontalMouseOffset = (scaledWidth/2) - (width/2 + horizontalOffset);      // centered
    verticalMouseOffset = (scaledHeight/2) - (height/2 + verticalOffset);         // centered
    //println(horizontalMouseOffset + " "+ verticalMouseOffset);
    //ellipse((mouseX + horizontalMouseOffset) * interfaceScale, (mouseY + verticalMouseOffset) * interfaceScale, 10, 10);
    if(balloon != null){
      balloon.setInterfaceScale(interfaceScale);
      balloon.setHorizontalOffset(horizontalMouseOffset);
      balloon.setVerticalOffset(verticalMouseOffset);
    }
  }
 
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.render(pgl, c.getXoffset(), c.getYoffset(), yflip);
    if(displayBoundingBoxes){
      textBlock.drawBoundingBox(c.getXoffset(), c.getYoffset());
    }
  }
  
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.move(c.getMWidth(), overflow);
  }
  
  if(mode == "authorcloud"){
    iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      textBlock.applyGravity(xGrav, yGrav, xgravity, ygravity);
    }
  }
  
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock instanceof QuoteLine){
      textBlock.checkCollisions(textBlocks, quoteMarginHorizontal, quoteMarginVertical);
    } else {
      textBlock.checkCollisions(textBlocks, textMarginHorizontal, textMarginVertical);
    }
  }
  
  // check all text blocks to see if they should be removed or not
  iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.remove){
      //println("removing "+ textBlock.getText());
      textBlocks.remove(textBlock.id);
      if(textBlock instanceof QuoteLine){
        quoteObjects.remove(textBlock.id);
        bioObjects.remove(textBlock.id);
        // apply attractor vector to surrounding authors to make them swarm back in
        float attractorVal = 2;
        int distance = 100;
        Iterator innerIter = textBlocks.values().iterator();
        while(innerIter.hasNext()){
          TextBlock otherBlock = (TextBlock)innerIter.next();
          //println(otherBlock.getText());
          if(textBlock.getID() != otherBlock.getID()){
            if((otherBlock.getY()+(otherBlock.getHeight()*0.5f) > textBlock.getY() - (textBlock.getHeight()*0.5f)) && (otherBlock.getY()-(otherBlock.getHeight()*0.5f) < textBlock.getY()+(textBlock.getHeight()*0.5f))){
              if(otherBlock.getX() > textBlock.getX()){
                //println((otherBlock.getX()-(otherBlock.getWidth()*0.5)) - (textBlock.getX()+(textBlock.getWidth()*0.5)));
                if((otherBlock.getX()-(otherBlock.getWidth()*0.5f)) - (textBlock.getX()+(textBlock.getWidth()*0.5f)) < distance){
                  //println("pulling on "+otherBlock.getText());
                  otherBlock.push(0 - attractorVal, 0);
                }
              } else {
                if((textBlock.getX()-(textBlock.getWidth()*0.5f)) - (otherBlock.getX()+(otherBlock.getWidth()*0.5f)) < distance){
                  //println("pulling on "+otherBlock.getText());
                  otherBlock.push(attractorVal, 0);
                }
              }
            }
          }
        }
      }
    }
  }
  
  // check all quote objects to see if they are fully off screen, and if so, remove them immediately
  iter = quoteObjects.values().iterator();
  while(iter.hasNext()){                                                // for every quote line...
    QuoteLine quoteLine = (QuoteLine)iter.next();
    if(quoteLine.getX()-(quoteLine.getWidth()/2) > client.getMWidth()){  // if quote line off screen...
      boolean removeQuote = true;
      Iterator innerIter = quoteObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherQuoteLine = (QuoteLine)innerIter.next();
        if(otherQuoteLine.getQuoteID() == quoteLine.getQuoteID()){      // if belonging to this quote..
          if(otherQuoteLine.getX()-(otherQuoteLine.getWidth()/2) < client.getMWidth()){  // if not beyond the right edge of the screen...
            removeQuote = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeQuote){
        quoteLine.removeNow();
        ((Author)authorObjects.get(quoteLine.getQuoteID())).deactivate();
      }
    } else if(quoteLine.getX() + (quoteLine.getWidth()/2) < 0){
      boolean removeQuote = true;
      Iterator innerIter = quoteObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherQuoteLine = (QuoteLine)innerIter.next();
        if(otherQuoteLine.getQuoteID() == quoteLine.getQuoteID()){      // if belonging to this quote..
          if(otherQuoteLine.getX() + (otherQuoteLine.getWidth()/2) > 0){
            removeQuote = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeQuote){
        quoteLine.removeNow();
        ((Author)authorObjects.get(quoteLine.getQuoteID())).deactivate();
      }
    }
  }
  
  // check all BIOGRAPHY objects to see if they are fully off screen, and if so, remove them immediately
  iter = bioObjects.values().iterator();
  while(iter.hasNext()){
    QuoteLine bioLine = (QuoteLine)iter.next();
    if(bioLine.getX()-(bioLine.getWidth()/2) > client.getMWidth()){ // if bio line off the screen...
      boolean removeBio = true;
      Iterator innerIter = bioObjects.values().iterator();
      while(innerIter.hasNext()){
        QuoteLine otherBioLine = (QuoteLine)innerIter.next();
        if(otherBioLine.getQuoteID() == bioLine.getQuoteID()){      // if belonging to this quote..
          if(otherBioLine.getX()-(otherBioLine.getWidth()/2) < client.getMWidth()){  // if not beyond the right edge of the screen...
            removeBio = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeBio){
        bioLine.removeNow();
        ((Author)authorObjects.get(bioLine.getQuoteID())).deactivate();
      }
    } else if(bioLine.getX() + (bioLine.getWidth()/2) < 0){
      boolean removeBio = true;
      Iterator innerIter = bioObjects.values().iterator();
      while(innerIter.hasNext()){                                        // check other quote lines belonging to this quote
        QuoteLine otherBioLine = (QuoteLine)innerIter.next();
        if(otherBioLine.getQuoteID() == bioLine.getQuoteID()){      // if belonging to this quote..
          if(otherBioLine.getX() + (otherBioLine.getWidth()/2) > 0){
            removeBio = false;                                        // remove nothing and break inner loop
            break;
          }
        }
      }
      if(removeBio){
        bioLine.removeNow();
        ((Author)authorObjects.get(bioLine.getQuoteID())).deactivate();
      }
    }
  }
  
  // CHECK ALL TEXTBLOCKS AGAINST RAMPMASK TO SEE IF THEY SHOULD BE PUSHED AWAY
  if(enableRampForce){
    rampMask.checkCollisions(textBlocks);
  }
  if(displayRamp){
    rampMask.draw();
  }
  
  // GRADIENTS ON EDGES TO FADE ALL TEXT HORIZONTALLY
  image(leftFade, 0, -1000, horizontalFallOff, client.getMHeight()+2000);
  image(rightFade, client.getMWidth() - horizontalFallOff, -1000, horizontalFallOff, client.getMHeight()+2000);
  // additional black area beyond screen just in case of scaling rounding issues
  fill(0);
  rect(-20,-1000,21,client.getMHeight()+2000);
  rect(client.getMWidth()-1,-1000,20,client.getMHeight()+2000);
  
  if(displayFrames){
    stroke(255);
    noFill();
    rect(0,156,1024,768);       // three projection frames
    rect(1024,156,1024,768); 
    rect(2048,156,1024,768);
    stroke(255,0,0);
    rect(0,0,3072,1080);        // entire MPE stage
    line(1536, 500, 1536, 580); // cross hairs in center of stage
    line(1496, 540, 1576, 540);
    noStroke();
  }
  
  if(enableCamera){
    popMatrix();
  }
  
  popMatrix();
  
  if(displayControls){
    //image(imageLanguageInstructionsImage, imageLanguageInstructionsX, imageLanguageInstructionsY);
    //image(imageViewingInstructionsImage, imageViewingInstructionsX, imageViewingInstructionsY);
    //image(imageZoomTitleImage, imageZoomTitleX, imageZoomTitleY);
    //image(imageTitleImage, imageTitleX, imageTitleY);
    //image(buttonSliderImage, buttonSliderX, buttonSliderY);  // TEMPORARY until widget object is created
    widgetManager.draw();
  }
  
  textFont(statFont);
  fill(200);
  if(displayStats){
    pushMatrix();
    translate(client.getXoffset(), client.getYoffset());
    text(PApplet.parseInt(frameRate) +" fps", 10, 20);
    
    if(System.currentTimeMillis() - lastTime > 0){
      float mpeDelay = 1000.0f / (System.currentTimeMillis() - lastTime);  // 1 second divided by duration since last frame in ms
      mpeFpsHistory.add(mpeDelay);
    }
    if(mpeFpsHistory.size() == 30){
      float mpeTotal = 0;
      for(int i=0; i<mpeFpsHistory.size(); i++){
        mpeTotal += (Float)mpeFpsHistory.get(i);
      }
      mpeFps = PApplet.parseInt(mpeTotal/mpeFpsHistory.size());
      mpeFpsHistory.remove(0);
    }
    
    text(mpeFps + " mpe fps", 10, 35);
    text(textBlocks.size() + " total text objects", 10, 50);
    text(authorObjects.size() +" author names", 10, 65);
    text(quoteObjects.size() +" quote lines", 10, 80);
    text(bioObjects.size() +" bio lines", 10, 95);
    lastTime = System.currentTimeMillis();  // record time for FPS comparison
    popMatrix();
  }
  
  inactivityCounter++;  // keep counting forever, gets reset on mousePressedEvent
  if(inactivityCounter > screensaverInactivity){
    screensaverActivated = true;
  } else {
    screensaverActivated = false;
    resetCounter = 0;
    zoomDelayCounter = 0;
  }
  
  if(screensaverActivated){
    iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock textBlock = (TextBlock)iter.next();
      textBlock.xv += screensaverSpeed;
    }
    
    /*
    if(zoomDelayCounter < inactivityZoomDelay){
      zoomDelayCounter++;
    } else {
      if(interfaceScale != defaultInterfaceScale && !zooming){
        zoomDuration = inactivityZoomDuration;
        zoomTarget = defaultInterfaceScale;
        zoomStart = interfaceScale;
        zoomCounter = 0;
        zooming = true;
      }
    }
    */
    
    if(!zooming && interfaceScale != defaultInterfaceScale){
      zoomDelayCounter++;
    }
    if(zoomDelayCounter > inactivityZoomDelay){
      if(interfaceScale != defaultInterfaceScale){
        //println("auto zooming activated");
        zoomDuration = inactivityZoomDuration;
        zoomTarget = defaultInterfaceScale;
        zoomStart = interfaceScale;
        zoomDelayCounter = 0;
        zooming = true;
      }
    }
    
    resetCounter++;
    if(resetCounter > resetDelay){
      // trigger "freak out" and re-arrange all author names as well as randomize and tween to a new textscale
      resetCloud();
      resetCounter = 0;
    }
  }
  
}

public void draw(){
  //if(standAlone){
  //  render(client);
  //}
}

public void frameEvent(TCPClient c){
  // TODO: this will take over for draw() when MPE is implemented.
  if(c.messageAvailable()){
    String[] msg = c.getDataMessage();
    //println(msg[0]);
    for(int i=0; i<msg.length; i++){
      String[] command = msg[i].split(",");
      if(command[0].equals("drag")){    // used for roll overs and moving the text cloud side to side
        Iterator iter = textBlocks.values().iterator();
        while(iter.hasNext()){
          TextBlock textBlock = (TextBlock)iter.next();
          textBlock.xv += Integer.parseInt(command[1])*dragDamp;
          float distance = abs(textBlock.y - Integer.parseInt(command[3]));
          if(distance < dragRadius){
            textBlock.xv += Integer.parseInt(command[1]) * (dragDamp * (1 - (distance/dragRadius)));
          }
          if(textBlock.isOver(Integer.parseInt(command[2]), Integer.parseInt(command[3]))){
            textBlock.rollOver();
          } else {
            textBlock.rollOut();
          }
        }
        lastDragged = System.currentTimeMillis();
      } else if(command[0].equals("press")){    // used to register intentional contact with author names
        mousePressedEvent(Integer.parseInt(command[1]), Integer.parseInt(command[2]));
      } else if(command[0].equals("release")){  // used to check for intentional click of author names
        //println("mouse release event received");
        mouseReleasedEvent(Integer.parseInt(command[1]), Integer.parseInt(command[2]));
      } else if(command[0].equals("buttonEvent")){
        if(command[1].equals("english")){
          userLanguage = "English";
          if(displayControls){
            btnEspanol.silentOff();
            if(balloon != null){
              balloon.setUserLanguage(userLanguage);
            }
          }
        } else if(command[1].equals("espanol")){
          userLanguage = "Espanol";
          if(displayControls){
            btnEnglish.silentOff();
            if(balloon != null){
              balloon.setUserLanguage(userLanguage);
            }
          }
        } else if(command[1].equals("zoomin")){
          if(enableCamera){
            if(interfaceScale > maxZoom){
              // TODO: tween the interfaceScale down by 0.1
              zoomDuration = buttonZoomDuration;
              zoomTarget = interfaceScale - 0.1f;
              zoomStart = interfaceScale;
              zooming = true;
              zoomCounter = 0;
            } else {
              btnPlus.silentOff();
            }
          }
        } else if(command[1].equals("zoomout")){
          if(enableCamera){
            if(interfaceScale < minZoom){
              // TODO: tween the interfaceScale up by 0.1
              zoomDuration = buttonZoomDuration;
              zoomTarget = interfaceScale + 0.1f;
              zoomStart = interfaceScale;
              zooming = true;
              zoomCounter = 0;
            } else {
              btnMinus.silentOff();
            }
          }
        } else if(command[1].equals("slide")){
          if(enableCamera){
            float offset = 0 - (Float.parseFloat(command[2]) - 0.5f);  // cloud is centered
            horizontalOffset = offset * scaledWidth;
            println("horizontalOffset: "+ horizontalOffset +" horizontalMouseOffset: "+ horizontalMouseOffset +" offset: "+ offset);
            //println(offset +" horizontalOffset: "+ horizontalOffset + " horizontalMouseOffset: "+ horizontalMouseOffset);
          }
        } else if(command[1].equals("cloud")){
          if(command[2].equals("normal")){
            if(displayControls){
              btnDate.silentOff();
              dropdownGenre.silentOff();
              btnPopularity.silentOff();
            }
            mode = "authorcloud";
            // TODO: cause all author names to condense towards the middle of the screen
            authorCloudMode();
          } else if(command[2].equals("date")){
            if(displayControls){
              dropdownGenre.silentOff();
              btnAuthorCloud.silentOff();
              btnPopularity.silentOff();
            }
            mode = "date";
            sortByDate();
            // TODO: PREVENT DRAGGING/SLIDING
          } else if(command[2].equals("genre")){
            if(displayControls){
              btnDate.silentOff();
              btnAuthorCloud.silentOff();
              btnPopularity.silentOff();
            }
            mode = "genre";
            //println("Displaying authors with "+ genreList_english.get(Integer.parseInt(command[3])) +" genre");
            sortByGenre((String)genreList_english.get(Integer.parseInt(command[3])));
            // TODO: PREVENT DRAGGING/SLIDING?
          } else if(command[2].equals("popularity")){
            if(displayControls){
              btnDate.silentOff();
              btnAuthorCloud.silentOff();
              dropdownGenre.silentOff();
            }
            mode = "popularity";
            sortByPopularity();
            // TODO: PREVENT DRAGGING/SLIDING
          } 
        }
      } else if(command[0].equals("quote")){    // affect the quote in some way
        if(command[1].equals("fadein")){
          // fade out any bio belonging to this author that exists first
          Iterator iter = bioObjects.values().iterator();
          while(iter.hasNext()){
            QuoteLine bioLine = (QuoteLine)iter.next();
            if(bioLine.getQuoteID() == Integer.parseInt(command[2])){
              bioLine.fadeOutAndRemove();
            }
          }
          Author author = (Author)textBlocks.get(Integer.parseInt(command[2]));
          if(author.hasQuote(userLanguage)){
            author.addControls(widgetManager, balloon);
            author.retrigger();
            createQuote(author);
          }
        }
      } else if(command[0].equals("bio")){      // affect the bio in some way
        if(command[1].equals("fadein")){
          // fade out any quote belonging to this author that exists first
          Iterator iter = quoteObjects.values().iterator();
          while(iter.hasNext()){
            QuoteLine quoteLine = (QuoteLine)iter.next();
            if(quoteLine.getQuoteID() == Integer.parseInt(command[2])){
              quoteLine.fadeOutAndRemove();
            }
          }
          Author author = (Author)textBlocks.get(Integer.parseInt(command[2]));
          if(author.hasBio(userLanguage)){
            author.addControls(widgetManager, balloon);
            author.retrigger();
            createBio(author);
          }
        }
      }
    }
  }
  render(c);
}











// WIDGET BASED CONTROL FUNCTIONS

public void redValue(float r){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setRed(r);
  }
}
public void greenValue(float g){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setGreen(g);
  }
}
public void blueValue(float b){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setBlue(b);
  }
}
public void alphaMax(float a){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setAlphaMax(a);
  }
}
public void alphaFallOff(float a){
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    textBlock.setAlphaFallOff(a);
  }
}
public void dragDamping(float d){
  dragDamp = d;
}





// USER INTERACTION FUNCTIONS

public void mousePressed(){
  if(displayControls){
    widgetManager.pressed(mouseX, mouseY);
    //client.broadcast("press," + (mouseX+(client.getXoffset()*2)) + "," + mouseY);
    //println("sending " + int(mouseX * 1.6) +","+ int(mouseY * 1.6 - 320));
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      //client.broadcast("press,"+ int(mouseX * interfaceScale) +","+ int((mouseY * interfaceScale) - verticalOffset));
      client.broadcast("press,"+ PApplet.parseInt((mouseX  + horizontalMouseOffset) * interfaceScale)+","+PApplet.parseInt((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

public void mousePressedEvent(int xpos, int ypos){
  //println("mouse pressed: "+xpos +" "+ypos);
  boolean clickedBackground = true;
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.isOver(xpos, ypos)){
      textBlock.press();
      clickedBackground = false;
    }
  }
  
  if(clickedBackground && displayControls){
    widgetManager.removeItem(balloon);
  }
  
  inactivityCounter = 0;  // keep track of last user action and count frames
}

public void mouseReleased(){
  //println("mouse released");
  if(displayControls){
    widgetManager.released(mouseX, mouseY);
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      //client.broadcast("release,"+ int(mouseX * interfaceScale) +","+ int((mouseY * interfaceScale) - verticalOffset));
      client.broadcast("release,"+ PApplet.parseInt((mouseX  + horizontalMouseOffset) * interfaceScale)+","+PApplet.parseInt((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

public void mouseReleasedEvent(int xpos, int ypos){
  //println("mouse released: "+xpos +" "+ypos);
  Iterator iter = textBlocks.values().iterator();
  while(iter.hasNext()){
    TextBlock textBlock = (TextBlock)iter.next();
    if(textBlock.isOver(xpos, ypos)){
      if(textBlock instanceof Author){
        //if(System.currentTimeMillis() - lastDragged > postDragDelay){
          if(((Author)textBlock).hasQuote(userLanguage)){
            if(!((Author)textBlock).triggered && textBlock.pressed){
              createQuote((Author)textBlock);  // create quote here based on author properties
              if(displayControls){
                widgetManager.removeItem(balloon);
                balloon = new Balloon("Balloon", (Author)textBlock, 0, balloonImage, balloonDown, buttonQuoteImage, buttonQuoteDown, buttonQuoteEspImage, buttonQuoteEspDown, buttonQuoteGrey, buttonBiographyImage, buttonBiographyDown, buttonBiographyEspImage, buttonBiographyEspDown, buttonBiographyGrey, interfaceScale, horizontalMouseOffset, verticalMouseOffset, userLanguage);
                widgetManager.addItem(balloon);
                ((Author)textBlock).addControls(widgetManager, balloon);
              }
              textBlock.release();
            } else {
              textBlock.releasedOutside();
            }
          }
          //textBlock.release();
        //} else {
        //  textBlock.releasedOutside();
        //}
      }
    } else {
      textBlock.releasedOutside();
    }
  }
}

public void mouseDragged(){
  if(displayControls){
    widgetManager.dragged(mouseX, mouseY);
    if(!widgetManager.isOverAWidget(mouseX, mouseY)){
      int force = PApplet.parseInt(mouseX*interfaceScale) - PApplet.parseInt(pmouseX*interfaceScale);
      //client.broadcast("drag,"+ force +","+ int(mouseX*interfaceScale) +","+ ((int)(mouseY*interfaceScale) - verticalOffset));
       client.broadcast("drag,"+ force +","+ PApplet.parseInt((mouseX  + horizontalMouseOffset) * interfaceScale)+","+PApplet.parseInt((mouseY + verticalMouseOffset) * interfaceScale));
    }
  }
}

public void keyPressed(){
  if(key == 'c' || key == 'C'){         // toggle gravity along the x axis
    if(displayCursor){
      noCursor();
    } else {
      cursor(ARROW);
    }
    displayCursor = !displayCursor;
  } else if(key == 'b' || key == 'B'){  // toggle bounding box display
    displayBoundingBoxes = !displayBoundingBoxes;
  } else if(key == 's' || key == 'S'){  // toggle statistics display
    displayStats = !displayStats;
  } else if(key == 'v' || key == 'V'){  // toggle camera feed
    enableCamera = !enableCamera;
  } else if(key == 'f' || key == 'F'){
    displayFrames = !displayFrames;
  } else if(key == 'y' || key == 'Y'){
    yflip = !yflip;
  } else if(key == 'r' || key == 'R'){
    displayRamp = !displayRamp;
  }
}
public class Author extends TextBlock {
  
  // AUTHOR.pde
  // the visual representation as well as metadata about the author.
  
  private String quote = "";
  public boolean triggered = false;
  private boolean rolledOver = false;
  private float targetTextScale, targetOpacity, defaultAlphaVal;
  private float selectedRedVal;
  private float selectedGreenVal;
  private float selectedBlueVal;
  //private int fadeInDuration;
  //private int holdDuration;
  //private int fadeOutDuration;
  private WidgetManager widgetManager;
  private Balloon balloon;  // reference to controls
  private float xtarget, ytarget;  // used for damped to target movement, such as genre/date/popularity mode
  private float xsource, ysource;
  private boolean dampedMovement = false;
  private int movementCounter = 0;
  private int movementDuration;
  private float maxOverlap;  // used to drop alpha when overlapping quotes
  private float textScaleTarget;
  private int textScaleDuration;
  private int textScaleCounter = 0;
  private boolean tweenScale = false;
  private boolean tweenDamping = false;
  private float targetXdamping, targetYdamping;
  private float originalXdamping, originalYdamping;
  private float authorFadeOutDelay;
  
  public String seminalwork_english, seminalwork_spanish;
  public String bio_english, bio_spanish;
  public String quote_english, quote_spanish;
  public String[] genres;
  public String[] geographies;
  public int born, died, workbegan, workended, popularity;
  
  public Author(int id, String name, float x, float y, String fontName, int fontSize, float textScale){
    super(id, name, x, y, fontName, fontSize, textScale);
    originalXdamping = xdamping;
    originalYdamping = ydamping;
    targetXdamping = 0.8f;
    targetYdamping = 0.8f;
  }
  
  public Author(int id, String name, String quote, float x, float y, String fontName, int fontSize, float textScale){
    super(id, name, x, y, fontName, fontSize, textScale);
    this.quote = quote;
    originalXdamping = xdamping;
    originalYdamping = ydamping;
    targetXdamping = 0.8f;
    targetYdamping = 0.8f;
  }
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    if(!triggered){
      super.applyGravity(xGrav, yGrav, xgravity, ygravity);
    }
  }
  
  public void addControls(WidgetManager widgetManager, Balloon balloon){
    this.widgetManager = widgetManager;
    this.balloon = balloon;
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    // only be able to slide horizontally.
    // only push against other text blocks, never get pushed.
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    maxOverlap = 1;  // start at 1, go as low as 0
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5f + b.getWidth()*0.5f) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5f + b.getHeight()*0.5f) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5f)+xMargin + (b.getWidth()*0.5f)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5f)+yMargin + (b.getHeight()*0.5f)+yMargin) - abs(y - b.getY());
          
          if(b instanceof QuoteLine){
            // compare area of overlap to total area to get percentage alpha should be
            float overlapPercentage = (xoverlap * yoverlap) / (w * h);
            maxOverlap -= overlapPercentage;
            if(maxOverlap < 0){
              maxOverlap = 0;
            }
          }
          
          /*
          // check if opposing textblock is a quote, and change alpha value based on amount of overlap
          if(b instanceof QuoteLine){
            //float xdist = abs(((QuoteLine)b).getParagraphCenterX() - x) / (((QuoteLine)b).getParagraphWidth()/2);
            //float ydist = abs(((QuoteLine)b).getParagraphCenterY() - y) / (((QuoteLine)b).getParagraphHeight()/2);
            float xdist = abs(((QuoteLine)b).getParagraphCenterX() - x) / (((QuoteLine)b).getParagraphWidth()/2);
            float ydist = abs(((QuoteLine)b).getParagraphCenterY() - y) / (((QuoteLine)b).getParagraphHeight()/2);
            float paragraphHypo = sqrt(sq(((QuoteLine)b).getParagraphWidth()) + sq(((QuoteLine)b).getParagraphHeight()));
            //maxOverlap = sqrt((xdist*xdist) + (ydist*ydist));
            //println("xdist: " + xdist + " ydist: "+ ydist);
            if(maxOverlap > 1){
              maxOverlap = 1;
            }
            println(maxOverlap);
          }
          */
          
          /*
          float opposingAlpha = b.getAlpha() / 255;
          //float totalOverlap = ((w - xoverlap) / w) * ((h - yoverlap) / h);
          float totalOverlap = ((w - xoverlap) / w);
          if(b instanceof QuoteLine){
            if(totalOverlap * opposingAlpha < maxOverlap){
              maxOverlap = totalOverlap * opposingAlpha;
              println(maxOverlap);
            }
          }
          */
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              if(!triggered){
                yv += yoverlap * 0.5f * verticalSpring * b.getScale();
              }
              b.push(0, 0 - yoverlap * 0.5f * verticalSpring * textScale);
            } else {
              if(!triggered){
                yv += xoverlap * 0.5f * verticalSpring * b.getScale();
              }
              b.push(0, 0 - xoverlap * 0.5f * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              if(!triggered){
                yv -= yoverlap * 0.5f * verticalSpring * b.getScale();
              }
              b.push(0, yoverlap * 0.5f * verticalSpring * textScale);
            } else {
              if(!triggered){
                yv -= xoverlap * 0.5f * verticalSpring * b.getScale();
              }
              b.push(0, xoverlap * 0.5f * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
              if(!triggered){
                xv += yoverlap * 0.5f * horizontalSpring * b.getScale();
              }
              b.push(0 - yoverlap * 0.5f * horizontalSpring * textScale, 0);
            } else {
              if(!triggered){
                xv += xoverlap * 0.5f * horizontalSpring * b.getScale();
              }
              b.push(0 - xoverlap * 0.5f * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              if(!triggered){
                xv -= yoverlap * 0.5f * horizontalSpring * b.getScale();
              }
              b.push(yoverlap * 0.5f * horizontalSpring * textScale, 0);
            } else {
              if(!triggered){
                xv -= xoverlap * 0.5f * horizontalSpring * b.getScale();
              }
              b.push(xoverlap * 0.5f * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void move(int stageWidth, int overflow){
    if(dampedMovement){
      if(movementCounter < movementDuration){
        float progress = sin((movementCounter / (float)movementDuration) * (PI/2));
        x = xsource + ((xtarget-xsource) * progress);
        y = ysource + ((ytarget-ysource) * progress);
        movementCounter++;
      } else {
        dampedMovement = false;
        movementCounter = 0;
        tweenDamping = true;
        xdamping = targetXdamping;
        ydamping = targetYdamping;
      }
    } else {
      if(tweenDamping){
        if(movementCounter < movementDuration){
          float progress = (movementCounter / (float)movementDuration);
          xdamping = targetXdamping - ((targetXdamping - originalXdamping) * progress);
          ydamping = targetYdamping - ((targetYdamping - originalYdamping) * progress);
          movementCounter++;
        } else {
          xdamping = originalXdamping;
          ydamping = originalYdamping;
          movementCounter = 0;
          tweenDamping = false;
        }
      }
      x += xv;
      if(!triggered){
        y += yv;
      }
      xv *= xdamping;
      yv *= ydamping;
      if(x-(w*0.5f) > stageWidth + overflow){    // wrap text block to other side of the screen
        x = (0-overflow)+10;
      } else if(x+(w*0.5f) < -overflow) {
        x = stageWidth+overflow-10;
      }
    }
  }
  
  public void moveTo(float xtarget, float ytarget, int movementDuration){
    this.xtarget = xtarget;
    this.ytarget = ytarget;
    xsource = x;
    ysource = y;
    this.movementDuration = movementDuration;
    dampedMovement = true;
  }
  
  public void scaleTo(float textScaleTarget, int textScaleDuration){
    println(textScaleTarget +" "+textScaleDuration);
    this.textScaleTarget = textScaleTarget;
    this.textScaleDuration = textScaleDuration;
    textScaleCounter = 0;
    tweenScale = true;
  }
  
  public boolean hasQuote(String language){
    if(language == "English"){
      if(quote_english.length() > 0){
        return true;
      }
    } else {
      if(quote_spanish.length() > 0){
        return true;
      }
    }
    return false;
  }
  
  public boolean hasBio(String language){
    if(language == "English"){
      if(bio_english.length() > 0){
        return true;
      }
    } else {
      if(bio_spanish.length() > 0){
        return true;
      }
    }
    return false;
  }
  
  public String getQuote(){
    return quote;
  }
  
  public void retrigger(){
    fadeOut = false;
    fadeIn = false;
    hold = true;
    counter = 0;
  }
  
  public void deactivate(){
    if(!fadeOut){
      fadeOut = true;
      fadeIn = false;
      hold = false;
      counter = 0;
    }
  }
  
  public void release(){
    //if(pressed && !triggered && hasQuote(userLanguage)){
    if(pressed && !triggered){
      triggered = true;
      fadeIn = true;
      counter = 0;
    }
    if(rolledOver || scaleDown){
      //println(textValue + " released");
      rolledOver = false;
      scaleDown = false;
      scaleUp = true;
      counter = 0;
    }
    super.release();
  }
  
  public void releasedOutside(){
    //if(rolledOver){
      //println(textValue + " released outside");
      rolledOver = false;
      scaleDown = false;
      scaleUp = true;
      counter = 0;
    //}
    super.releasedOutside();
  }
  
  public void rollOver(){
    if(!rolledOver && !triggered){
      //println(textValue + " rolled over");
      scaleDown = true;
      rolledOver = true;
      targetTextScale = defaultTextScale * rollOverScale;
      targetOpacity = alpha(c) * rollOverAlpha;
      counter = 0;
    }
  }
  
  public void rollOut(){
    if(rolledOver){
      //println(textValue + " rolled out");
      scaleDown = false;
      rolledOver = false;
      scaleUp = true;
      counter = 0;
    }
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    int hh = stageHeight/2;
    if(y > hh){
      defaultAlphaVal = alphaStartVal - ((y-hh)/hh)*alphaFallOffVal;
    } else {
      defaultAlphaVal = alphaStartVal - ((hh-y)/hh)*alphaFallOffVal;
    }
    
    if(tweenScale){
      textScaleCounter++;
      float progress = sin((textScaleCounter / (float)textScaleDuration) * (PI/2));
      if(textScaleCounter > textScaleDuration){
        defaultTextScale = textScale = textScaleTarget;
        println(textScale);
        tweenScale = false;
        textScaleCounter = 0;
      }
      textScale = defaultTextScale + ((textScaleTarget - defaultTextScale) * progress);
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    }
    
    // apply overlap value to transparency
    defaultAlphaVal *= maxOverlap;
    //println("overlap %: "+maxOverlap +" default alpha: "+defaultAlphaVal);
    
    if(scaleDown){                                     // ROLLED OVER
      float progress = sin((counter / (float)rollOverDuration) * (PI/2));
      textScale = defaultTextScale + (targetTextScale - defaultTextScale) * progress;
      alphaVal = defaultAlphaVal - (defaultAlphaVal - targetOpacity) * progress;
      if(counter == rollOverDuration){
        scaleDown = false;
        textScale = targetTextScale;
        alphaVal = targetOpacity;
      }
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    } else if(scaleUp && textScale < defaultTextScale){       // ROLLED OUT
      float progress = sin((counter / (float)rollOutDuration) * (PI/2));
      textScale = targetTextScale + (defaultTextScale - targetTextScale) * progress;
      alphaVal = targetOpacity + (defaultAlphaVal - targetOpacity) * progress;
      if(counter == rollOutDuration){
        scaleUp = false;
        textScale = defaultTextScale;
        alphaVal = defaultAlphaVal;
      }
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    } else if(rolledOver){                                     // HOLDING OVER
      // holding while cursor over
    } else {                                                   // REGULAR MODE
      alphaVal = defaultAlphaVal;
    }
    
    /*
    // THIS DOES HORIZONTAL ALPHA FADING
    if(x-(w/2) < horizontalFallOff){
      alphaVal = defaultAlphaVal * ((float)(x-(w/2))/horizontalFallOff);
    } else if(x+(w/2) > stageWidth-horizontalFallOff){
      alphaVal = defaultAlphaVal * ((float)(stageWidth - (x+(w/2)))/horizontalFallOff);
    } else if(x-(w/2) < 0 || x+(w/2) > stageWidth){
      alphaVal = 0;
    } else {
      alphaVal = defaultAlphaVal;
    }
    */
    
    if(triggered){
      if(fadeIn){
        float progress = sin((counter / (float)fadeInDuration) * (PI/2));
        if(counter == fadeInDuration){
          fadeIn = false;
          hold = true;
          counter = 0;
        }
        
        if(defaultTextScale < maxTextScale){  // if less than max text scale...
          textScale = defaultTextScale + ((maxTextScale - defaultTextScale) * progress);
          w = textRender.getWidth(this.textValue) * textScale;    // necessary to keep it centered
          h = textRender.getHeight(this.textValue) * textScale;
        }
        
        c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, (255 - alphaVal)*progress+ alphaVal);
        //c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      } else if(hold){
        float progress = counter / (float)holdDuration;
        if(counter >= holdDuration){
          hold = false;
          fadeOut = true;
          counter = 0;
        } else if(counter >= (holdDuration - authorFadeOutDelay) + fadeOutDuration){
          if(widgetManager != null && balloon != null){
            widgetManager.removeItem(balloon);
          }
        } else if(counter >= holdDuration - authorFadeOutDelay){
          if(balloon != null){
            balloon.fadeOut(fadeOutDuration);
          }
        }
        c = color(selectedRedVal, selectedGreenVal, selectedBlueVal, 255);
        //c = color(selectedRedVal, selectedGreenVal, selectedBlueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      } else if(fadeOut){
        float progress = 1 - sin((counter / (float)fadeOutDuration) * (PI/2));
        
        if(defaultTextScale < maxTextScale){  // if less than max text scale...
          textScale = defaultTextScale + ((maxTextScale - defaultTextScale) * progress);
          w = textRender.getWidth(textValue) * textScale;
          h = textRender.getHeight(textValue) * textScale;
        }
        
        if(counter == fadeOutDuration){
          fadeOut = false;
          triggered = false;
          counter = 0;
          //if(widgetManager != null && balloon != null){
          //  widgetManager.removeItem(balloon);
          //}
        }
        
        c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, (255 - alphaVal)*progress+ alphaVal);
        //c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      }
    } else {
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    }
    counter++;
  }
  
  public void softenCollisions(){
    movementCounter = 0;
    tweenDamping = true;
    xdamping = targetXdamping;
    ydamping = targetYdamping;
  }
  
  public void setFadeInDuration(int fadeInDuration){
    this.fadeInDuration = fadeInDuration;
  }
  
  public void setHoldDuration(int holdDuration){
    this.holdDuration = holdDuration;
  }
  
  public void setFadeOutDuration(int fadeOutDuration){
    this.fadeOutDuration = fadeOutDuration;
  }
  
  public void setAuthorFadeOutDelay(int authorFadeOutDelay){
    this.authorFadeOutDelay = authorFadeOutDelay;  // additional delay after quote fades out
  }
  
  public void setSelectedRed(float selectedRedVal){
    this.selectedRedVal = selectedRedVal;
  }
  
  public void setSelectedGreen(float selectedGreenVal){
    this.selectedGreenVal = selectedGreenVal;
  }
  
  public void setSelectedBlue(float selectedBlueVal){
    this.selectedBlueVal = selectedBlueVal;
  }
  
  public void push(float xforce, float yforce){
    if(!triggered){
      yv += yforce;
      xv += xforce;
    }
  }
  
}
public class Balloon extends Widget implements WidgetListener{
  
  private PImage img, imgDown;
  private PImage quoteImg, quoteDown, quoteImgEsp, quoteDownEsp, quoteGrey, bioImg, bioDown, bioImgEsp, bioDownEsp, bioGrey;
  private Button quote;
  private Button bio;
  public Author author;
  public boolean quoteMode = true;
  public boolean bioMode = false;
  private boolean fadeout = false;
  private int counter = 0;
  private int fadeOutDuration;
  private float horizontalOffset, verticalOffset;
  private float interfaceScale;
  private boolean flipped = false;
  private String userLanguage;
  
  public Balloon(String name, Author author, float value, PImage img, PImage imgDown, PImage quoteImg, PImage quoteDown, PImage quoteImgEsp, PImage quoteDownEsp, PImage quoteGrey, PImage bioImg, PImage bioDown, PImage bioImgEsp, PImage bioDownEsp, PImage bioGrey, float interfaceScale, float horizontalOffset, float verticalOffset, String userLanguage){
    super(name, PApplet.parseInt(author.getX()*(1/interfaceScale)), PApplet.parseInt((author.getY()+verticalOffset)*(1/interfaceScale)) - img.height, value);
    this.author = author;
    this.w = img.width;
    this.h = img.height;
    this.img = img;
    this.imgDown = imgDown;
    this.quoteImg = quoteImg;
    this.quoteDown = quoteDown;
    this.quoteImgEsp = quoteImgEsp;
    this.quoteDownEsp = quoteDownEsp;
    this.quoteGrey = quoteGrey;
    this.bioImg = bioImg;
    this.bioDown = bioDown;
    this.bioImgEsp = bioImgEsp;
    this.bioDownEsp = bioDownEsp;
    this.bioGrey = bioGrey;
    this.interfaceScale = interfaceScale;
    this.horizontalOffset = horizontalOffset;
    this.verticalOffset = verticalOffset;
    this.userLanguage = userLanguage;
    
    x = PApplet.parseInt(((author.getX()-(author.getWidth()/2)) / interfaceScale) - horizontalOffset) + 10 - img.width;
    y = PApplet.parseInt((author.getY() / interfaceScale) - verticalOffset) + 10 - img.height;
    int xpos = 0;
    if(x < 0){
      flipped = true;
      xpos = img.width - quoteImg.width;
    }
    
    if(userLanguage == "English"){
      if(author.hasQuote(userLanguage)){
        quote = new Button("Quote", xpos, 0, 0, quoteImg, quoteDown);
        quote.silentOn();
      } else {
        quote = new Button("Quote", xpos, 0, 0, quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio = new Button("Biography", xpos, 48, 0, bioImg, bioDown);
      } else {
        bio = new Button("Biography", xpos, 48, 0, bioGrey, bioGrey);
      }
    } else {
      if(author.hasQuote(userLanguage)){
        quote = new Button("Quote", xpos, 0, 0, quoteImgEsp, quoteDownEsp);
        quote.silentOn();
      } else {
        quote = new Button("Quote", xpos, 0, 0, quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio = new Button("Biography", xpos, 48, 0, bioImgEsp, bioDownEsp);
      } else {
        bio = new Button("Biography", xpos, 48, 0, bioGrey, bioGrey);
      }
    }
    
    //quote = new Button("Quote", xpos, 0, 0, quoteImg, quoteDown);
    quote.addListener(this);
    //bio = new Button("Biography", xpos, 48, 0, bioImg, bioDown);
    bio.addListener(this);
  }
  
  public void draw(){
    pushMatrix();
    
    if(flipped){
      x = PApplet.parseInt(((author.getX()+(author.getWidth()/2)) / interfaceScale) - horizontalOffset);
    } else {
      x = PApplet.parseInt(((author.getX()-(author.getWidth()/2)) / interfaceScale) - horizontalOffset) + 10 - img.width;
    }
    y = PApplet.parseInt((author.getY() / interfaceScale) - verticalOffset) + 10 - img.height;
    
    translate(x, y);
    if(fadeout){
      float progress = sin((counter / (float)fadeOutDuration) * (PI/2));
      tint(255, 255 - (progress*255));
      counter++;
    }
    if(flipped){
      pushMatrix();
      scale(-1,1,1);
      if(quoteMode){
        image(img,0-img.width,0);
      } else {
        image(imgDown,0-img.width,0);
      }
      popMatrix();
    } else {
      if(quoteMode){
        image(img,0,0);
      } else {
        image(imgDown,0,0);
      }
    }
    quote.draw();
    bio.draw();
    if(fadeout){
      tint(255, 255);
    }
    popMatrix();
  }
  
  public void dragged(){
    // probably need to inform drop down items from here
  }
  
  public void fadeOut(int duration){
    fadeout = true;
    fadeOutDuration = duration;
  }
  
  public void pressed(){
    if(!fadeout){
      quote.mousePressed(mouseX-x, mouseY-y);
      bio.mousePressed(mouseX-x, mouseY-y);
    }
  }
  
  public void released(){
  }
  
  public void rollOver(){
    //quote.rollOver();
    //bio.rollOver();
  }
  
  public void rollOut(){
    //quote.rollOut();
    //bio.rollOut();
  }
  
  public void cursorMovement(){
    //quote.cursorMovement();
    //bio.cursorMovement();
  }
  
  public void setAuthor(Author author){
    this.author = author;
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    if(we.widget == quote){
      bio.silentOff();
      quoteMode = true;
      bioMode = false;
      // TODO: trigger quote reveal here
    } else {
      quote.silentOff();
      bioMode = true;
      quoteMode = false;
      // TODO: trigger biography reveal here
    }
    super.newEvent(new WidgetEvent(this, PRESSED, true));
  }
  
  public void setInterfaceScale(float interfaceScale){
    this.interfaceScale = interfaceScale;
  }
  
  public void setHorizontalOffset(float horizontalOffset){
    this.horizontalOffset = horizontalOffset;
  }
  
  public void setVerticalOffset(float verticalOffset){
    this.verticalOffset = verticalOffset;
  }
  
  public void setUserLanguage(String userLanguage){
    this.userLanguage = userLanguage;
    if(userLanguage == "English"){
      if(author.hasQuote(userLanguage)){
        quote.switchImages(quoteImg, quoteDown);
      } else {
        quote.switchImages(quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio.switchImages(bioImg, bioDown);
      } else {
        bio.switchImages(bioGrey, bioGrey);
      }
    } else {
      if(author.hasQuote(userLanguage)){
        quote.switchImages(quoteImgEsp, quoteDownEsp);
      } else {
        quote.switchImages(quoteGrey, quoteGrey);
      }
      if(author.hasBio(userLanguage)){
        bio.switchImages(bioImgEsp, bioDownEsp);
      } else {
        bio.switchImages(bioGrey, bioGrey);
      }
    }
  }
  
}
public class Button extends Widget{
  
  private boolean on = false;
  private PImage img;
  private PImage imgDown;
  private boolean imgMode = false;
  private VTextRenderer textRender;
  private float textScale;
  private float textWidth, textHeight;
  
  public Button(String name, int x, int y, int w, int h, float value){
    super(name, x, y, w, h, value);
    String fontName = "HelveticaNeue BoldCond";
    int fontSize    = 16;
    textRender = new VTextRenderer(fontName, fontSize);
    textRender.setColor( 1, 1, 1, 1 );
    textWidth = textRender.getWidth(name);
    textHeight = textRender.getHeight(name);
  }
  
  public Button(String name, int x, int y, int w, int h, float value, PImage img){
    super(name, x, y, w, h, value);
    this.img = img;
    imgMode = true;
  }
  
  public Button(String name, int x, int y, float value, PImage img, PImage imgDown){
    super(name, x, y, value);
    this.img = img;
    this.imgDown = imgDown;
    imgMode = true;
    w = img.width;
    h = img.height;
  }
  
  public void draw(){
    pushMatrix();
    if(imgMode){
      translate(x,y);
      if(on){
        image(imgDown,0,0);
      } else {
        image(img,0,0);
      }
    } else {
      noStroke();
      if(mouseOver && on){
        fill(activeForegroundColor);
      } else if(mouseOver && !on){
        fill(foregroundColor);
      } else if(on){
        fill(activeColor);
      } else {
        fill(backgroundColor);
      }
      rect(0,y,w,h);
      // this should be better
      textRender.print(name, PApplet.parseInt(x+(w/2)) - PApplet.parseInt(textWidth/2), (height - 50 - y) - PApplet.parseInt(h/2 + textHeight/2));
      
    }
    popMatrix();
  }
  
  public void silentToggle(){
    on = !on;
  }
  
  public void silentOff(){
    on = false;
  }
  
  public void silentOn(){
    on = true;
  }
  
  public void pressed() {
    if(!on){
      //on = !on;
      on = true;
      WidgetEvent we = new WidgetEvent(this, PRESSED, on);
      super.newEvent(we);
    }
  }

  public void released() {
  }

  public void rollOut() {
    println("rolled out of "+name);
  }

  public void rollOver() {
    println("rolled over "+name);
  }

  public void cursorMovement() {
  }

  public void dragged() {
    // TODO Auto-generated method stub	
  }	
  
  public void switchImages(PImage img, PImage imgDown){
    this.img = img;
    this.imgDown = imgDown;
  }
  
}
public class DropDownMenu extends Widget implements WidgetListener{
  
  // DROPDOWNMENU.pde
  // encapsulates many buttons and controls when they're displayed/accessable.
  // also manages the receipt of widget events from buttons and broadcasts them
  // back to the widget manager.
  
  private int buttonWidth, buttonHeight, leading;
  private ArrayList items;
  private boolean displayItems = false;
  private boolean on = false;
  private PImage img, imgDown;
  
  public DropDownMenu(String name, int x, int y, float value, int buttonHeight, int leading, PImage img, PImage imgDown){
    super(name, x, y, value);
    this.w = img.width;
    this.h = img.height;
    this.buttonWidth = w;
    this.buttonHeight = buttonHeight;
    this.leading = leading;
    this.img = img;
    this.imgDown = imgDown;
    this.items = new ArrayList();
  }
  
  public void addItem(String name, float val){
    Button b = new Button(name, x, y + (buttonHeight+leading)*items.size(), buttonWidth, buttonHeight, val);
    b.addListener(this);
    items.add(b);
    //super.h = ((buttonHeight+leading)*items.size())-leading;
  }
  
  public void draw(){
    pushMatrix();
    translate(x, y);
    if(on){
      image(imgDown,0,0);
    } else {
      image(img,0,0);
    }
    if(displayItems){
      Iterator i = items.iterator();
      while(i.hasNext()){
        Button b = (Button)i.next();
        b.draw();
      }
    }
    popMatrix();
  }
  
  public void dragged(){
    // probably need to inform drop down items from here
  }
  
  public void pressed(){
    if(displayItems){
      Iterator i = items.iterator();
      while(i.hasNext()){
        ((Button)i.next()).mousePressed(mouseX-img.height, mouseY-img.height);
      }
      displayItems = false;
      h = img.height;
    } else {
      displayItems = true;
      h = img.height + ((buttonHeight+leading)*items.size())-leading;
    }
  }
  
  public void released(){
  }
  
  public void rollOver(){
    // highlight
  }
  
  public void rollOut(){
    displayItems = false;
  }
  
  public void silentOff(){
    on = false;
  }
  
  public void cursorMovement(){
    // must report mouse at all times, not just when cursor enters/exits radio button area
    Iterator i = items.iterator();
    while(i.hasNext()){
      //((Button)i.next()).mouseMoved(mouseX-x, mouseY-y);
      ((Button)i.next()).mouseMoved(mouseX-img.height, mouseY-img.height);
    }
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    ((Button)we.widget).silentOff();  // prevent button from staying highlighted
    WidgetEvent newwe = new WidgetEvent(this, PRESSED, true);
    super.newEvent(newwe);
    on = true;
  }
  
}
public class QuoteLine extends TextBlock{
  
  // QUOTELINE.pde
  // works in clusters to override conventional animation/collision detection.
  
  private int quoteID;
  private int lineNumber;
  private Author author;           // author this quote came from
  private int introDelayDuration;  // duration to wait for textblocks to get pushed out of the way before appearing
  private int fadeInDuration;
  private int holdDuration;
  private int fadeOutDuration;
  private long startTime;
  private float trueWidth, trueHeight;
  private float quotationOffset;
  private float defaultAlphaVal = 255;
  private boolean alignRight = false;  // default aligned left
  private float pushMultiplier = 0.5f;
  private float centerX, centerY;
  private float paragraphWidth, paragraphHeight;
  private int leftMargin = 0;
  
  public QuoteLine(int id, int quoteID, int lineNumber, Author author, String textValue, float x, float y, String fontName, int fontSize, float textScale, float quotationOffset){
    super(id, textValue, x, y, fontName, fontSize, textScale);
    this.quoteID = quoteID;
    this.lineNumber = lineNumber;
    this.author = author;
    this.quotationOffset = quotationOffset;
    uppercase = false;
    xv = 0;
    yv = 0;
    if(!uppercase){
      trueWidth = w = textRender.getWidth(this.lowercaseTextValue) * textScale;
      trueHeight = h = textRender.getHeight(this.lowercaseTextValue) * textScale;
    }
    introDelay = true;
    counter = 0;
  }
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    // ignore gravity
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    // only be able to slide horizontally.
    // only push against other text blocks, never get pushed.
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5f + b.getWidth()*0.5f) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5f + b.getHeight()*0.5f) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5f)+xMargin + (b.getWidth()*0.5f)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5f)+yMargin + (b.getHeight()*0.5f)+yMargin) - abs(y - b.getY());
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              //yv += yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - yoverlap * pushMultiplier * verticalSpring * textScale);
            } else {
              //yv += xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - xoverlap * pushMultiplier * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              //yv -= yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, yoverlap * pushMultiplier * verticalSpring * textScale);
            } else {
              //yv -= xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, xoverlap * pushMultiplier * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
             //xv += yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - yoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            } else {
              //xv += xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - xoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              //xv -= yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(yoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            } else {
              //xv -= xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(xoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void clearArea(ConcurrentHashMap textBlocks, float multiplier){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){
        if((abs(x - b.getX()) < abs(w*0.5f + b.getWidth()*0.5f)) && (abs(y - b.getY()) < abs(h*0.5f + b.getHeight()*0.5f))){
          if(b.getX() > x){
            float xoverlap = (x+(w*0.5f)) - (b.getX()-(b.getWidth()*0.5f));
            b.push((xoverlap / (w*0.5f)) * multiplier, 0);
          } else {
            float xoverlap = (b.getX() + (b.getWidth()*0.5f)) - (x-(w*0.5f));
            b.push((xoverlap / (w*0.5f)) * multiplier, 0);
          }
        }
      } 
    }
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    
    if(introDelay){
      float progress = counter / (float)introDelayDuration;
      //w = trueWidth * progress;
      if(counter == introDelayDuration){
        introDelay = false;
        fadeIn = true;
        counter = 0;
      }
    } else if(fadeIn){
      float progress = sin((counter / (float)fadeInDuration) * (PI/2));
      if(counter == fadeInDuration){
        fadeIn = false;
        hold = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, alphaVal*progress);
      
      pushMatrix();
      if(yflip){
        translate(x-xoffset, (stageHeight-y) - yoffset, 0);
      } else {
        translate(x-xoffset, y-yoffset, 0);
      }
      rotateX(radians(180));// fix for text getting rendered upside down for some reason
      //c = color(redVal, greenVal, blueVal, alphaVal);
      textRender.setColor( red(c)/255, green(c)/255, blue(c)/255, alpha(c)/255);
      pgl.beginGL();
      if(uppercase){
        textRender.print(uppercaseTextValue, 0-(w*0.5f), 0-(h*0.5f), 0, textScale);
        //textRender.print( join(subset(uppercaseTextValue.split(""), 0 , numChars), ""), 0-(w*0.5),0-(h*0.5),0,textScale);
      } else {
        textRender.print(lowercaseTextValue, 0-(w*0.5f), 0-(h*0.5f), 0, textScale);
        //textRender.print( join(subset(lowercaseTextValue.split(""), 0 , numChars), ""), 0-(w*0.5),0-(h*0.5),0,textScale);
      }
      pgl.endGL();
      popMatrix();
      
    } else if(hold){
      float progress = counter / (float)holdDuration;
      //println(counter +" "+ holdDuration);
      if(counter == holdDuration){
        hold = false;
        fadeOut = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    } else if(fadeOut){
      float progress = sin((counter / (float)fadeOutDuration) * (PI/2));
      if(counter == fadeOutDuration){
        fadeOut = false;
        remove = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, 255 - (alphaVal*progress));
      super.render(pgl, xoffset, yoffset, yflip);
    } else {
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    }
    
    if(!fadeOut && !alignRight){
      x = (((author.getX() - (author.getWidth()*0.5f)) + (w*0.5f)) + 1 - quotationOffset);  // stay left aligned with author name at all times
    } else if(!fadeOut && alignRight){
      x = ((author.getX() + (author.getWidth()*0.5f)) + (w*0.5f)) + leftMargin;
      y = (author.getY() + (author.getHeight()*0.5f)) - (h*0.5f);
    }
    
    counter++;
  }
  
  public void push(float xforce, float yforce){
    // ignore vertical pushes
    if(!introDelay){
      //xv += xforce; // ignore horizontal pushes until typing animation
    }
  }
  
  public Author getAuthor(){
    return author;
  }
  
  public void snapToRight(int leftMargin){
    alignRight = true;
    this.leftMargin = leftMargin;
  }
  
  public int getQuoteID(){
    return quoteID;
  }
  
  public int getLineNumber(){
    return lineNumber;
  }
  
  public void setIntroDelay(int introDelayDuration){
    this.introDelayDuration = introDelayDuration;
  }
  
  public void setFadeInDuration(int fadeInDuration){
    this.fadeInDuration = fadeInDuration;
  }
  
  public void setHoldDuration(int holdDuration){
    this.holdDuration = holdDuration;
  }
  
  public void setFadeOutDuration(int fadeOutDuration){
    this.fadeOutDuration = fadeOutDuration;
  }
  
  public void setPushMultiplier(float pushMultiplier){
    this.pushMultiplier = pushMultiplier;
  }
  
  public void setParagraphCenter(float centerX, float centerY){
    this.centerX = centerX;
    this.centerY = centerY;
  }
  
  public void setParagraphDimensions(float paragraphWidth, float paragraphHeight){
    this.paragraphWidth = paragraphWidth;
    this.paragraphHeight = paragraphHeight;
  }
  
  public float getParagraphCenterX(){
    return centerX;
  }
  
  public float getParagraphCenterY(){
    return centerY;
  }
  
  public float getParagraphWidth(){
    return paragraphWidth;
  }
  
  public float getParagraphHeight(){
    return paragraphHeight;
  }
  
}
public class RampMask{
  
  // RAMPMASK.pde
  // used to mask off projection over the ramp area, as well as push textblocks away.
  
  private int x1, y1;  // top left
  private int x2, y2;  // rop right
  private int x3, y3;  // bottom right
  private int x4, y4;  // bottom left
  private int rampTopLeft[] = new int[2];     // top left
  private int rampTopRight[] = new int[2];    // top right
  private int rampBottomRight[] = new int[2];  // bottom right
  private int rampBottomLeft[] = new int[2];  // bottom left
  private int rampCenter[] = new int[2];       // ramp center
  
  public RampMask(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4){
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.x3 = x3;
    this.y3 = y3;
    this.x4 = x4;
    this.y4 = y4;
    
    rampTopLeft[0] = x1;
    rampTopLeft[1] = y1;
    rampTopRight[0] = x2;
    rampTopRight[1] = y2;
    rampBottomRight[0] = x3;
    rampBottomRight[1] = y3;
    rampBottomLeft[0] = x4;
    rampBottomLeft[1] = y4;
    rampCenter[0] = x1 + (x2 - x1)/2;
    rampCenter[1] = y1 + (y3 - (y1 + ((y2 - y1)/2)))/2;  // compensate for ramp of top edge
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks){
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      /*
      // THIS ONLY AFFECTS BLOCKS WITH BOTTOM EDGE DIRECTLY INTERSECTING RAMP SLOPE
      // check bottom edge of text against ramp sloping edge
      int[] vec1 = new int[2];    // bottom left
      vec1[0] = int(b.getX() - b.getWidth()/2);
      vec1[1] = int(b.getY() + b.getHeight()/2);
      int[] vec2 = new int[2];    // bottom right
      vec2[0] = int(b.getX() + b.getWidth()/2);
      vec2[1] = int(b.getY() + b.getHeight()/2);
      if(edgeIntersection(vec1, vec2, rampVec1, rampVec2)){
        //println(b.getText() + " intersecting ramp");
        b.push(0, -1);  // arbitrary upward force
      }
      */
      
      // if the line from any of textblocks corners to the center of the ramp DOESN'T cross any of the ramps edges...
      // then it must be inside the ramp polygon, and the textblock needs to be pushed away
      int[] topleft = new int[2];
      topleft[0] = PApplet.parseInt(b.getX() - b.getWidth()/2);
      topleft[1] = PApplet.parseInt(b.getY() - b.getHeight()/2);
      int[] topright = new int[2];
      topright[0] = PApplet.parseInt(b.getX() + b.getWidth()/2);
      topright[1] = PApplet.parseInt(b.getY() - b.getHeight()/2);
      int[] bottomleft = new int[2];
      bottomleft[0] = PApplet.parseInt(b.getX() - b.getWidth()/2);
      bottomleft[1] = PApplet.parseInt(b.getY() + b.getHeight()/2);
      int[] bottomright = new int[2];
      bottomright[0] = PApplet.parseInt(b.getX() + b.getWidth()/2);
      bottomright[1] = PApplet.parseInt(b.getY() + b.getHeight()/2);
      boolean inside = false;
      
      // check topleft corner against all edges
      if(!edgeIntersection(topleft, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(topleft, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(topleft, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(topleft, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check topright corner against all edges
      if(!edgeIntersection(topright, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(topright, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(topright, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(topright, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check bottomleft corner against all edges
      if(!edgeIntersection(bottomleft, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(bottomleft, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(bottomleft, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(bottomleft, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      // check bottomright corner against all edges
      if(!edgeIntersection(bottomright, rampCenter, rampTopLeft, rampTopRight) &&
         !edgeIntersection(bottomright, rampCenter, rampTopRight, rampBottomRight) &&
         !edgeIntersection(bottomright, rampCenter, rampBottomRight, rampBottomLeft) &&
         !edgeIntersection(bottomright, rampCenter, rampTopLeft, rampBottomRight)){
         inside = true;
      }
      
      if(inside){
        b.push(0,-1);
      }
      
    }
  }
  
  // this came from http://gpwiki.org/index.php/Polygon_Collision
  private double determinant(int[] vec1, int[] vec2){
    return vec1[0]*vec2[1]-vec1[1]*vec2[0];
  }
 
  //one edge is a-b, the other is c-d
  public boolean edgeIntersection(int[] a, int[] b, int[] c, int[] d){
    double det=determinant(subtractVector(b,a),subtractVector(c,d));
    double t=determinant(subtractVector(c,a),subtractVector(c,d))/det;
    double u=determinant(subtractVector(b,a),subtractVector(c,a))/det;
    if ((t<0)||(u<0)||(t>1)||(u>1)){
      return false;
    }
    //return a*(1-t)+t*b;
    return true;
  }
  
  public int[] subtractVector(int[] a, int[] b){
    int[] newvec = new int[2];
    newvec[0] = a[0] - b[0];
    newvec[1] = a[1] - b[1];
    return newvec;
  }
  
  public void draw(){  // only drawn on projection version
    noStroke();
    fill(0);
    quad(x1, y1, x2, y2, x3, y3, x4, y4);
  }
  
}
public class Slider extends Widget implements WidgetListener{
  
  private PImage img;
  public SliderBar sliderBar;
  private int sliderXtarget, sliderXsource;
  private int barSlideDuration;
  private int barSlideCounter;
  private boolean barSliding = false;
  
  public Slider(String name, int x, int y, float value, PImage img, PImage segmentImg, PImage segmentImgDown,
                PImage leftImg, PImage leftImgDown, PImage rightImg, PImage rightImgDown){
    super(name, x, y, img.width, img.height, value);
    this.img = img;
    sliderBar = new SliderBar("SliderBar",0,0,w,h,value, segmentImg, segmentImgDown, leftImg, leftImgDown, rightImg, rightImgDown);
    sliderBar.addListener(this);
  }
  
  public void dragged(){
    sliderBar.mouseDragged(mouseX-x, mouseY-y);
  }
  
  public void draw(){
    pushMatrix();
    translate(x,y,0);
    image(img,0,0);
    sliderBar.draw();
    if(barSliding){
      float progress = sin((barSlideCounter / (float)barSlideDuration) * (PI/2));
      float xpos = sliderXsource + ((sliderXtarget - sliderXsource) * progress);
      sliderBar.setOffset(PApplet.parseInt(xpos));
      WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
      super.newEvent(newwe);
      barSlideCounter++;
      if(barSlideCounter == barSlideDuration){
        barSliding = false;
        barSlideCounter = 0;
        sliderBar.setOffset(sliderXtarget);
        super.newEvent(new WidgetEvent(this, DRAGGED, true));
      }
    }
    popMatrix();
  }
  
  public void pressed(){
    if(!sliderBar.mouseInside(mouseX-x, mouseY-y)){
      if(mouseX-x < sliderBar.getX()){
        sliderXsource = sliderBar.getX();
        sliderXtarget = mouseX-x;
        barSlideCounter = 0;
        barSliding = true;
        //sliderBar.setOffset(mouseX-x);
        //WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
        //super.newEvent(newwe);
      } else if(mouseX-x > sliderBar.getX()+sliderBar.getWidth()){
        sliderXsource = sliderBar.getX();
        sliderXtarget = (mouseX-x) - sliderBar.getWidth();
        barSlideCounter = 0;
        barSliding = true;
        //sliderBar.setOffset((mouseX-x) - sliderBar.getWidth());
        //WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
        //super.newEvent(newwe);
      }
    } else {
      sliderBar.mousePressed(mouseX-x, mouseY-y);
    }
  }
  
  public void released(){
    sliderBar.mouseReleased(mouseX-x, mouseY-y);
  }
  
  public void rollOut(){
    // nothing?
  }
  
  public void rollOver(){
    // nothing?
  }
  
  public void cursorMovement(){
    sliderBar.mouseMoved(mouseX-x, mouseY-y);
  }
  
  public void setAreaVisible(float areaVisible){
    // normalized value used to shrink the width of the bar
    sliderBar.setWidth(PApplet.parseInt(w*areaVisible));
  }
  
  public void setOffset(float offset){
    // normalized value used to move the bar
    sliderBar.setOffset(PApplet.parseInt(w*offset));
  }
  
  public void setBarSlideDuration(int barSlideDuration){
    this.barSlideDuration = barSlideDuration;
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
    super.newEvent(newwe);
  }
  
  public float getBarPosition(){
    //return sliderBar.getX() / (float)w;  // normalized location (left aligned)
    return (sliderBar.getX() + (sliderBar.getWidth()*0.5f)) / (float)w;  // normalized location (centered)
  }
  
}
public class SliderBar extends Widget{
  
  private int clickX;
  private int totalWidth;
  private boolean dragging = false;
  private PImage segmentImg, leftImg, rightImg;
  private PImage segmentImgDown, leftImgDown, rightImgDown;
  
  public SliderBar(String name, int x, int y, int w, int h, float value, PImage segmentImg, PImage segmentImgDown,
                   PImage leftImg, PImage leftImgDown, PImage rightImg, PImage rightImgDown){
    super(name, x, y, w, h, value);
    this.totalWidth = w;
    this.segmentImg = segmentImg;
    this.leftImg = leftImg;
    this.rightImg = rightImg;
    this.segmentImgDown = segmentImgDown;
    this.leftImgDown = leftImgDown;
    this.rightImgDown = rightImgDown;
  }
  
  public void dragged(){
    // TODO: move bar location along slider and continually send widgetEvents giving normalized offset value
    if(x >= 0 && x+w <= totalWidth){
      x = mouseX + clickX;
      if(x < 0){
        x = 0;
      } else if(x+w > totalWidth){
        x = totalWidth - w;
      }
      WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
      super.newEvent(newwe);
    } 
  }
  
  public void draw(){
    //noStroke();
    //fill(50,150,255,128);
    //rect(x,y,w,h);
    if(dragging){
      image(leftImgDown, x, y);
      image(segmentImgDown, x + leftImg.width, y, PApplet.parseInt(w - (leftImg.width+rightImg.width)) + 1, segmentImg.height);
      image(rightImgDown, PApplet.parseInt(x + leftImg.width + (w - (leftImg.width+rightImg.width))), y);
    } else {
      image(leftImg, x, y);
      image(segmentImg, x + leftImg.width, y, PApplet.parseInt(w - (leftImg.width+rightImg.width)) + 1, segmentImg.height);
      image(rightImg, PApplet.parseInt(x + leftImg.width + (w - (leftImg.width+rightImg.width))), y);
    }
  }
  
  public void pressed(){
    clickX = x - mouseX;
    dragging = true;
  }
  
  public void released(){
    dragging = false;
  }
  
  public void rollOut(){
    // ignore
  }
  
  public void rollOver(){
    // ignore
  }
  
  public void cursorMovement(){
    // ignore
  }
  
  public void setWidth(int w){
    this.w = w;
    if(x < 0){
      x = 0;
    } else if(x+w > totalWidth){
      x = totalWidth - w;
    }
  }
  
  public void setOffset(int offset){
    this.x = offset;
    if(x < 0){
      x = 0;
    } else if(x+w > totalWidth){
      x = totalWidth - w;
    }
  }
  
}
public class TextBlock {
  
  // TEXTBLOCK.pde
  // Controls the visual appearance and collision functionality for text on screen.
  // Extended by Author and Quote classes.
  
  // TEMPORAL VARIABLES
  protected int id;
  protected float x, y;
  protected float w, h;
  protected float redVal             = 0;
  protected float greenVal           = 150;
  protected float blueVal            = 255;
  protected float alphaVal           = 255;
  protected float alphaStartVal      = 255;
  protected float alphaFallOffVal    = 255;
  protected float horizontalFallOff  = 200;
  protected int c;
  
  // TEXT VARIABLES
  protected String textValue, uppercaseTextValue, lowercaseTextValue;
  protected float textScale;
  protected float defaultTextScale;
  protected float maxTextScale;
  protected VTextRenderer textRender;
  protected String fontName;
  protected int fontSize;
  protected float xMargin, yMargin;
  
  // MOVEMENT VARIABLES
  protected float xv = random(-1.0f,1.0f);  // x velocity
  protected float yv = random(-1.0f,1.0f);  // y velocity
  protected float xdamping = 0.97f;        // horizontal motion damping
  protected float ydamping = 0.97f;        // vertical motion damping
  private float gAngle;                   // Angle to gravity center in degrees
  private float gTheta;                   // Angle to gravity center in radians
  private float gxv;                      // Gravity velocity along x axis
  private float gyv;                      // Gravity velocity along y axis
  protected float horizontalSpring;       // multiplier for amount of movement away from each other
  protected float verticalSpring;
  protected int stageWidth, stageHeight;
  protected int counter;
  
  // MODE VARIABLES
  protected boolean fadeIn         = false;
  protected boolean hold           = false;
  protected boolean fadeOut        = false;
  protected boolean scaleUp        = false;
  protected boolean scaleDown      = false;
  protected boolean rollOverEffect = false;
  protected boolean introDelay     = false;
  protected boolean uppercase      = true;
  public boolean remove            = false;
  
  // EFFECT VARIABLES
  protected int rollOverDuration;
  protected int rollOutDuration;
  protected float rollOverScale;
  protected float rollOverAlpha;
  protected int fadeInDuration;
  protected int holdDuration;
  protected int fadeOutDuration;
  
  // INTERACTION VARIABLES
  protected boolean pressed = false;    // triggered true when mouseDown is detected while over this textblock
  
  
  
  
  
  public TextBlock(int id, String textValue, float x, float y, String fontName, int fontSize, float textScale){
    this.id = id;
    this.lowercaseTextValue = textValue;
    this.textValue = this.uppercaseTextValue = textValue.toUpperCase();
    uppercaseTextValue = textValue.toUpperCase();
    this.x = x;
    this.y = y;
    this.fontName = fontName;
    this.fontSize = fontSize;
    this.textScale = textScale;
    this.defaultTextScale = textScale;
    
    textRender = new VTextRenderer(fontName, fontSize);
    textRender.setColor( 1, 1, 1, 1 );
    w = textRender.getWidth(this.textValue) * textScale;
    h = textRender.getHeight(this.textValue) * textScale;
  }
  
  public TextBlock(){
    
  }
  
  
  
  
  
  
  // PHYSICAL ARRANGEMENT FUNCTIONS
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    gAngle        = -radians(findAngle(x,y,xGrav,yGrav));
    gxv           = cos(gAngle) * xgravity;
    gyv           = sin(gAngle) * ygravity;
    xv += gxv;
    yv += gyv;
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5f + b.getWidth()*0.5f) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5f + b.getHeight()*0.5f) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5f)+xMargin + (b.getWidth()*0.5f)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5f)+yMargin + (b.getHeight()*0.5f)+yMargin) - abs(y - b.getY());
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              yv += yoverlap * 0.5f * verticalSpring * b.getScale();
              b.push(0, 0 - yoverlap * 0.5f * verticalSpring * textScale);
            } else {
              yv += xoverlap * 0.5f * verticalSpring * b.getScale();
              b.push(0, 0 - xoverlap * 0.5f * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              yv -= yoverlap * 0.5f * verticalSpring * b.getScale();
              b.push(0, yoverlap * 0.5f * verticalSpring * textScale);
            } else {
              yv -= xoverlap * 0.5f * verticalSpring * b.getScale();
              b.push(0, xoverlap * 0.5f * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
              xv += yoverlap * 0.5f * horizontalSpring * b.getScale();
              b.push(0 - yoverlap * 0.5f * horizontalSpring * textScale, 0);
            } else {
              xv += xoverlap * 0.5f * horizontalSpring * b.getScale();
              b.push(0 - xoverlap * 0.5f * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              xv -= yoverlap * 0.5f * horizontalSpring * b.getScale();
              b.push(yoverlap * 0.5f * horizontalSpring * textScale, 0);
            } else {
              xv -= xoverlap * 0.5f * horizontalSpring * b.getScale();
              b.push(xoverlap * 0.5f * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void clearArea(ConcurrentHashMap textBlocks, float multiplier){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){
        if((abs(x - b.getX()) < abs(w*0.5f + b.getWidth()*0.5f)) && (abs(y - b.getY()) < abs(h*0.5f + b.getHeight()*0.5f))){
          if(b.getX() > x){
              float xoverlap = (x+(w*0.5f)) - (b.getX()-(b.getWidth()*0.5f));
              b.push((xoverlap / (w*0.5f)) * multiplier, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5f)) - (x-(w*0.5f));
              b.push((xoverlap / (w*0.5f)) * multiplier, 0);
            }
        }
      } 
    }
  }
  
  public void clearAbove(ConcurrentHashMap textBlocks, float distance){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){
        if(b.getY() < y && b.getY() >= y - distance){  // if "over" text block...
          if(b.getX()+(b.getWidth()*0.5f) > x-(w*0.5f) && b.getX()-(b.getWidth()*0.5f) < x+(w*0.5f)){
            //println(b.getText() +" moved from above "+textValue);
            if(b.getX() > x){
              float xoverlap = (x+(w*0.5f)) - (b.getX()-(b.getWidth()*0.5f));
              //b.xv += (xoverlap / (w*0.5)) * 5;    // TODO: get multiplier from properties
              b.push((xoverlap / (w*0.5f)) * 5, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5f)) - (x-(w*0.5f));
              //b.xv -= (xoverlap / (w*0.5)) * 5;
              b.push((xoverlap / (w*0.5f)) * 5, 0);
            }
          }
        }
      }
    }
  }
  
  public void clearUnderneath(ConcurrentHashMap textBlocks, float distance){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){
        if(b.getY() > y && b.getY() <= y + distance){  // if under text block...
          if(b.getX()+(b.getWidth()*0.5f) > x-(w*0.5f) && b.getX()-(b.getWidth()*0.5f) < x+(w*0.5f)){
            //println(b.getText() +" moved from underneath "+textValue);
            if(b.getX() > x){
              float xoverlap = (x+(w*0.5f)) - (b.getX()-(b.getWidth()*0.5f));
              //b.xv += (xoverlap / (w*0.5)) * 5;    // TODO: get multiplier from properties
              b.push((xoverlap / (w*0.5f)) * 5, 0);
            } else {
              float xoverlap = (b.getX() + (b.getWidth()*0.5f)) - (x-(w*0.5f));
              //b.xv -= (xoverlap / (w*0.5)) * 5;
              b.push((xoverlap / (w*0.5f)) * 5, 0);
            }
          }
        }
      }
    }
  }
  
  public void fadeOutAndRemove(){
    if(!fadeOut){
      hold = false;
      fadeIn = false;
      introDelay = false;
      fadeOut = true;
      counter = 0;
    }
  }
  
  public void fadeOutAndRemove(int frameDelay){
    if(!fadeOut){
      holdDuration = frameDelay;
      hold = true;
      fadeIn = false;
      introDelay = false;
      counter = 0;
    }
  }
  
  public void removeNow(){
    remove = true;
  }
  
  private float findDistance(float x1, float y1, float x2, float y2){
    float xd = x1 - x2;
    float yd = y1 - y2;
    float td = sqrt(xd * xd + yd * yd);
    return td;
  }
    
  private float findAngle(float x1, float y1, float x2, float y2){
    float xd = x1 - x2;
    float yd = y1 - y2;
  
    float t = atan2(yd,xd);
    float a = (180 + (-(180 * t) / PI));
    return a;
  }
  
  public void move(int stageWidth, int overflow){
    x += xv;
    y += yv;
    xv *= xdamping;
    yv *= ydamping;
    if(x-(w*0.5f) > stageWidth + overflow){    // wrap text block to other side of the screen
      x = (0-overflow)+10;
    } else if(x+(w*0.5f) < -overflow) {
      x = stageWidth+overflow-10;
    }
  }
  
  
  
  
  
  // RENDERING METHODS
  
  public void drawBoundingBox(int xoffset, int yoffset){
    pushMatrix();
    translate(x-xoffset, y-yoffset, 0);
    stroke(255);
    noFill();
    rect(0-(w*0.5f),0-(h*0.5f), w, h);  // bounding box around text
    stroke(100);
    rect(0-((w*0.5f) + xMargin),0-((h*0.5f) + yMargin), w+(xMargin*2), h+(yMargin*2));  // margin bounding box
    popMatrix();
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    pushMatrix();
    // THIS TRANSLATE REQUIRED FLIPPING WHEN SWITCHED TO MPE
    if(yflip){
      translate(x-xoffset, (stageHeight-y) - yoffset, 0);
    } else {
      translate(x-xoffset, y-yoffset, 0);
    }
    rotateX(radians(180));// fix for text getting rendered upside down for some reason
    textRender.setColor( red(c)/255, green(c)/255, blue(c)/255, alpha(c)/255);
    pgl.beginGL();
    if(uppercase){
      textRender.print( uppercaseTextValue, 0-(w*0.5f),0-(h*0.5f),0,textScale);
    } else {
      textRender.print( lowercaseTextValue, 0-(w*0.5f),0-(h*0.5f),0,textScale);
    }
    pgl.endGL();
    popMatrix();
  }
  
  
  
  
  
  // PROPERTY RETRIEVAL METHODS
  
  public float getX(){
    return x;
  }
  public float getY(){
    return y;
  }
  
  public float getXVel(){
    return xv;
  }
  public float getYVel(){
    return yv;
  }
  
  public float getWidth(){
    return w;
  }
  public float getHeight(){
    return h; 
  }
  
  // returns potential max dimension given maxTextScale
  public float getMaxWidth(){
    return textRender.getWidth(this.textValue) * maxTextScale;
  }
  public float getMaxHeight(){
    return textRender.getHeight(this.textValue) * maxTextScale;
  }
  
  public float getAlpha(){
    return alphaVal;
  }
  
  public float getScale(){
    return textScale;
  }
  
  public String getText(){
    return textValue;
  }
  
  public VTextRenderer getTextRenderer(){
    return textRender;
  }
  
  public int getID(){
    return id;
  }
  
  
  
  
  
  // INTERACTION METHODS
  
  public boolean isOver(int mx, int my){
    if(mx > x-(w/2) && mx < x+(w/2) && my > y-(h/2) && my < y+(h/2)){
      return true;
    }
    return false;
  }
  
  public void press(){
    pressed = true;
  }
  
  public void release(){
    pressed = false;
  }
  
  public void releasedOutside(){
    pressed = false;
  }
  
  public void rollOver(){
    // TODO: trigger rollOver effect
  }
  
  public void rollOut(){
    
  }
  
  
  
  
  
  // APPEARANCE/POSITION CHANGE METHODS
  
  public void push(float xforce, float yforce){
    xv += xforce;
    yv += yforce;
  }
  
  public void setDamping(float damping){
    this.xdamping = damping;
    this.ydamping = damping;
  }
  public void setHorizontalDamping(float xdamping){
    this.xdamping = xdamping;
  }
  public void setVerticalDamping(float ydamping){
    this.ydamping = ydamping;
  }
  
  public void setScale(float textScale){
    this.textScale = textScale;
    this.defaultTextScale = textScale;
    w = textRender.getWidth(this.textValue) * textScale;
    h = textRender.getHeight(this.textValue) * textScale;
  }
  
  public void setMaxScale(float maxTextScale){
    this.maxTextScale = maxTextScale;
  }
  
  public void setHorizontalSpring(float spring){
    horizontalSpring = spring;
  }
  public void setVerticalSpring(float spring){
    verticalSpring = spring;
  }
  
  public void setX(float x){
    this.x = x;
  }
  public void setY(float y){
    this.y = y;
  }
  
  public void setXMargin(float xMargin){
    this.xMargin = xMargin;
  }
  public void setYMargin(float yMargin){
    this.yMargin = yMargin;
  }
  
  public void setRollOverDuration(int rollOverDuration){
    this.rollOverDuration = rollOverDuration;
  }
  public void setRollOutDuration(int rollOutDuration){
    this.rollOutDuration = rollOutDuration;
  }
  
  public void setRollOverScale(float rollOverScale){
    this.rollOverScale = rollOverScale;
  }
  public void setRollOverAlpha(float rollOverAlpha){
    this.rollOverAlpha = rollOverAlpha;
  }
  
  public void setStageWidth(int stageWidth){
    this.stageWidth = stageWidth;
  }
  public void setStageHeight(int stageHeight){
    this.stageHeight = stageHeight;
  }
  
  
  
  // COLOR CONTROL METHODS FROM SLIDERS
  
  public void setRed(float redVal){  
    this.redVal = redVal;
  }
  public void setGreen(float greenVal){
    this.greenVal = greenVal;
  }
  public void setBlue(float blueVal){
    this.blueVal = blueVal;
  }
  public void setAlphaMax(float alphaStartVal){
    this.alphaStartVal = alphaStartVal;
  }
  public void setAlphaFallOff(float alphaFallOffVal){
    this.alphaFallOffVal = alphaFallOffVal;
  }
  public void setHorizontalFallOff(float horizontalFallOff){
    this.horizontalFallOff = horizontalFallOff;  // distance for alpha fade to the sides
  }
  
}





class VTextRenderer{
  
 int _w, _h;
 
 String _fontName;
 int _fontSize;
 TextRenderer _textRender;
 Font font;
 
 VTextRenderer( String fontName, int size )
 {
   _fontName = fontName;
   _fontSize = size;
   _textRender = new TextRenderer( new Font(fontName, Font.TRUETYPE_FONT, size), true, true, null, true );
   _textRender.setColor( 1.0f, 1.0f, 1.0f, 1.0f );
   //_textRender.setUseVertexArrays( true );
 }

 VTextRenderer( String fontName, int size, boolean antialiased, boolean mipmap )
 {
   _fontName = fontName;
   _fontSize = size;
   _textRender = new TextRenderer( new Font(fontName, Font.TRUETYPE_FONT, size), antialiased, true, null, mipmap );
   _textRender.setColor( 1.0f, 1.0f, 1.0f, 1.0f );
   //_textRender.setUseVertexArrays( true );
 }
 
 public float getHeight(String textLabel){
   Rectangle2D rectBounds = _textRender.getBounds(textLabel);
   return (float)rectBounds.getHeight();
 }
 
 public float getWidth(String textLabel){
   Rectangle2D rectBounds = _textRender.getBounds(textLabel);
   return (float)rectBounds.getWidth();
 }
 
 public void print( String str, int x, int y )
 {
   _textRender.beginRendering( width, height, true );
   _textRender.draw( str, x, y );
   _textRender.endRendering();  
   _textRender.flush();
 }
 
 public void printArea( String str, int x, int y, int w, int h )
 {
   _textRender.beginRendering( w, h, true );  // hopefully limits area being drawn to
   _textRender.draw( str, x, y );
   _textRender.endRendering();  
   _textRender.flush();
 }

 public void print( String str, float x, float y, float z )
 {
   print( str, x, y, z, 1.0f );
 }

 public void print( String str, float x, float y, float z, float s )
 {
   _textRender.begin3DRendering();
   _textRender.draw3D( str, x, y, z, s );
   _textRender.end3DRendering();  
   _textRender.flush();
 }

 public void setColor( float c )
 {
   setColor( c, c, c, 1 );
 }

 public void setColor( float c, float a )
 {
   setColor( c, c, c, a );
 }

 public void setColor( float r, float g, float b )
 {
   setColor( r, g, b, 1 );
 }
 
 public void setColor( float r, float g, float b, float a )
 {
   _textRender.setColor( r, g, b, a );
 }
 
 public void setSmoothing( boolean flag )
 {
   _textRender.setSmoothing( flag );
 }
 
}
public abstract class Widget {
  
  // WIDGET.pde
  // Used to capture all types of cursor interaction and act as a super class for all interface items.
  
  private ArrayList listeners;
  protected String name;
  protected int x, y, w, h;
  protected float value;
  protected boolean mouseOver           = false;
  protected boolean mouseDown           = false;
  protected int backgroundColor       = color(50, 50, 50, 200);
  protected int foregroundColor       = color(100, 100, 100, 200);
  protected int activeColor           = color(200, 0, 0, 255);
  protected int activeForegroundColor = color(255, 0, 0, 255);
  protected final int PRESSED = 0;
  protected final int RELEASED = 1;
  protected final int ROLLOVER = 2;
  protected final int ROLLOUT = 3;
  protected final int DRAGGED = 4;
  
  public Widget(String name, int x, int y, float value){
    this.name = name;
    this.x = x;
    this.y = y;
    this.value = value;
    listeners = new ArrayList();
  }
  
  public Widget(String name, int x, int y, int w, int h, float value){
    this.name = name;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.value = value;
    listeners = new ArrayList();
  } 
  
  final public void addListener(WidgetListener wl){
    listeners.add(wl);
  }
	
  final public void removeListener(WidgetListener wl){
    listeners.remove(wl);
  }
	
  final public void newEvent(WidgetEvent we){
    Iterator i = listeners.iterator();
    while (i.hasNext()){
      ((WidgetListener)i.next()).widgetEvent(we);
    }
  }
  
  
  /* MOUSE EVENT FUNCTIONS */
	
  public abstract void draw();
  public abstract void pressed();
  public abstract void released();
  public abstract void dragged();
  public abstract void rollOver();
  public abstract void rollOut();
  public abstract void cursorMovement();
  
  final public void mouseMoved(int mouseX, int mouseY){
    if(mouseInside(mouseX, mouseY)){	// if within constraints, activate rollOver
      if(!mouseOver){
        mouseOver = true;
        rollOver();
      }
    } else {				// if outside constraints, activate rollOut
      if(mouseOver){
        mouseOver = false;
        rollOut();
      }
    }
    cursorMovement();			// verbose movement repeater (needed for embedded items)
  }
	
  final public void mouseDragged(int mouseX, int mouseY){
    if(mouseDown){
      dragged();
    }
  }
	
  final public void mousePressed(int mouseX, int mouseY){
    if(mouseInside(mouseX, mouseY)){
      mouseDown = true;
      pressed();
    }
  }

  final public void mouseReleased(int mouseX, int mouseY){
    if(mouseDown){
      released();
      mouseDown = false;
    }
  }
	
  final public boolean mouseInside(int mouseX, int mouseY){
    if((mouseX >= x && mouseX <= x+w) && (mouseY >= y && mouseY <= y+h)){
      return true;
    } else {
      return false;
    }
  }
	
  final public String getName(){
    return name;
  }
  
  final public int getX(){
    return x;
  }
  
  final public int getY(){
    return y;
  }
  
  final public int getWidth(){
    return w;
  }
  
  final public int getHeight(){
    return h;
  }
  
  final public void setX(int x){
    this.x = x;
  }
  
  final public void setY(int y){
    this.y = y;
  }
  
  
  
  /* COLOR SETTER FUNCTIONS */
	
  final public void setBackgroundColor(int c){
    backgroundColor = c;
  }
	
  final public void setForegroundColor(int c){
    foregroundColor = c;
  }
	
  final public void setActiveColor(int c){
    activeColor = c;
  }
	
  final public void setActiveForegroundColor(int c){
    activeForegroundColor = c;
  }
	
  final public float getValue(){
    return value;
  }
  
}
public class WidgetEvent {
  
  // WIDGETEVENT.pde
  // holds properties on widget events to be passed to the manager.

  public Widget widget;			// reference to the control
  public String name;			// name of the control
  public int type;			// type of event
  public boolean state;			// state of the event
	
  public WidgetEvent(Widget widget, int type, boolean state){
    this.widget = widget;
    this.name = widget.name;
    this.type = type;
    this.state = state;
  }
	
}
public interface WidgetListener {
  public abstract void widgetEvent(WidgetEvent we);
}
public class WidgetManager implements WidgetListener {
  
  // WIDGETMANAGER.pde
  // gets called by PApplet core to communicate cursor interaction.
  // also receives widget events from widgets in order to determine the
  // next course of action.
  
  private TCPClient client;   // reference to mpe broadcaster
  private ArrayList widgets;  // root level widgets
 
  public WidgetManager(TCPClient client){
    this.client = client;
    widgets = new ArrayList();
  }
  
  public void addItem(Widget w){
    w.addListener(this);
    widgets.add(w);
  }
  
  public void removeItem(Widget w){
    if(widgets.contains(w)){
      w.removeListener(this);
      widgets.remove(w);
    }
  }
  
  public void draw(){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).draw();
    }
  }
  
  public boolean isOverAWidget(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      if(((Widget)i.next()).mouseInside(mouseX, mouseY)){
        return true;
      }
    }
    return false;
  }
  
  public void pressed(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mousePressed(mouseX, mouseY);
    }
  }
  
  public void released(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mouseReleased(mouseX, mouseY);
    }
  }
  
  public void dragged(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      Widget w = (Widget)i.next();
      w.mouseDragged(mouseX, mouseY);
      w.mouseMoved(mouseX, mouseY);
    }
  }
  
  public void cursorMovement(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mouseMoved(mouseX, mouseY);
    }
  }
 
  public void widgetEvent(WidgetEvent we){
    //println(we.name);
    if(we.name.equals("English")){
      client.broadcast("buttonEvent,english");
    } else if(we.name.equals("Espanol")){
      client.broadcast("buttonEvent,espanol");
    } else if(we.name.equals("Author Cloud")){
      client.broadcast("buttonEvent,cloud,normal");
    } else if(we.name.equals("By Date")){
      client.broadcast("buttonEvent,cloud,date");
    } else if(we.name.equals("By Popularity")){
      client.broadcast("buttonEvent,cloud,popularity");
    } else if(we.name.equals("By Genre")){
      client.broadcast("buttonEvent,cloud,genre,"+PApplet.parseInt(we.widget.value));
    } else if(we.name.equals("ZoomIn")){
      client.broadcast("buttonEvent,zoomin");
    } else if(we.name.equals("ZoomOut")){
      client.broadcast("buttonEvent,zoomout");
    } else if(we.name.equals("Balloon")){
      if(((Balloon)we.widget).quoteMode){
        client.broadcast("quote,fadein,"+((Balloon)we.widget).author.getID());
      } else {
        client.broadcast("bio,fadein,"+((Balloon)we.widget).author.getID());
      }      
    } else if(we.name.equals("Slider")){
      client.broadcast("buttonEvent,slide,"+((Slider)we.widget).getBarPosition());
    }
  } 
  
}

  static public void main(String args[]) {
    PApplet.main(new String[] { "--present", "--bgcolor=#666666", "--hide-stop", "FILMain" });
  }
}
