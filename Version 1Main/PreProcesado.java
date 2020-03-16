package Proyecto_Preguntas;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class PreProcesado 
{
	
	private String ruta;
	
	public PreProcesado( String pRuta ) { this.ruta = pRuta; }
	
	// CONVIERTE A FORMATO ARFF EL FICHERO DE REGUNTAS EN FORMATO CSV.
	public void getRawARFF( String pNombre ) 
	{
		String datoscsv = this.ruta + "/" + pNombre + ".csv"; 
		String rawarff = this.ruta + "/" + pNombre + "_RaW.arff";
		String tmp = ruta + "/ztmp.csv";
    	
    	try
        {
        	System.out.println("Iniciando conversion del fichero " + pNombre + "...");
        	
        	BufferedReader br = new BufferedReader(new FileReader( datoscsv ));
        	BufferedWriter bw = new BufferedWriter(new FileWriter( tmp ));
			
        	String linea;
        	while((linea = br.readLine()) != null)
        	{
        		// ELIMINAR CUALQUIER CARACTER NO ASCII DE LAS INSTANCIAS
        		linea = linea.replaceAll("[^\\p{ASCII}]", "");
        		linea = linea.replaceAll("\\.", "");
        		linea = linea.replaceAll("\\?", "");
        		linea = linea.replaceAll("\\:", "");
        		linea = linea.replaceAll("\\-", "");
        		linea = linea.replaceAll("\\&", "");
        		linea = linea.replaceAll("'", "");
        		linea = linea.replaceAll("`", "");
        			
        		// DIVIDIR EN TRES PARTES PARA SEPARAR LOS ATRIBUTOS
        		String[] partes = linea.split(",", 3);
        		
        		// ESCRIBIR EL RESULTADO
        		bw.write(partes[2]);
        		bw.write(" , ");
        		bw.write(partes[1]);
        		
                bw.newLine();		
        	}
        	br.close();
        	bw.close();
        	System.out.println("   Fichero temporal limpio...");
        	
        	// CARGAR EL CSV LIMPIO Y MARCAR COMO STRING LA PREGUNTA
           	CSVLoader loader = new CSVLoader();
            loader.setSource( new File( tmp ) );
            loader.setStringAttributes("last");
        	Instances data = loader.getDataSet();    	
        	System.out.println("   Fichero temporal cargado...");
        	
        	// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + pNombre);
        	System.out.println("   Nombre de la relacion establecida...");
        	
        	// GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
        	 BufferedWriter writer = new BufferedWriter(new FileWriter( rawarff ));
        	 writer.write(data.toString());
        	 writer.flush();
        	 writer.close();
        	 System.out.println("Conversion del fichero " + pNombre + " finalizada... \n");
        }
        catch(Exception e) { e.printStackTrace(); }
	}
	
	// APLICA LA VECTORIZACION BoW o TF-IDF COMPRIMIDO COMO Sparse o NonSparse
	public void getWordVector( String pNombre , String pBow , String pSparse ) 
	{
		String name = pNombre;
	
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }
    	
		String config = "";
		
		if ( bow ) { config += "_BoW_"; }
    	else { config += "_TFIDF_"; }
    	
    	if ( sparse ) { config += "Sparse"; }
    	else { config += "NonSparse"; }
    		    	
    	String rawarff = this.ruta + "/" + name + "_RaW.arff";
    	String bowarff = this.ruta + "/" + name + config + ".arff";   	
    	
		try
	    {
			System.out.println("Iniciando la conversion " + config + "...");
			
			// CARGAMOS EL CONJUNTO DE DATOS DEL RAW
			Instances data = new DataSource( rawarff ).getDataSet();
			
			// TOKENICER
			NGramTokenizer tokenizer = new NGramTokenizer();
			tokenizer.setNGramMaxSize(1);
			tokenizer.setNGramMinSize(1);
			tokenizer.setDelimiters( " \n 	.,;'\"()?!-/<>‘’“”…«»•&{[|`^]}$*%" );
		    
			// APLICAMOS EL FILTRO StringToWordVector
	    	StringToWordVector filtroBow = new StringToWordVector();
	    	filtroBow.setInputFormat( data );
	    	filtroBow.setOutputWordCounts( !bow );  
	    	filtroBow.setIDFTransform( !bow );
	    	filtroBow.setTFTransform( !bow );
	    	filtroBow.setLowerCaseTokens(true);
	    	filtroBow.setOutputWordCounts(true);
	    	filtroBow.setTokenizer( tokenizer );
	    	
	    	data = Filter.useFilter(data, filtroBow);
	    	System.out.println("   Filtro StringToWordVector aplicado...");
	    	
	    	// APLICAMOS EL FILTRO NonSparseToSparse
	    	if ( !sparse ) 
	    	{
	    		NonSparseToSparse filtroSparse = new NonSparseToSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("   Filtro NonSparseToSparsede aplicado...");
	    	}
	    	else
	    	{
	    		SparseToNonSparse filtroSparse = new SparseToNonSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("   Filtro NonSparseToSparsede aplicado...");
	    	}
	    	
	    	// MOVEMOS LA CLASE A PREDECIR AL FINAL
			Reorder filtroR = new Reorder();
			filtroR.setAttributeIndices("2-last,1");
			filtroR.setInputFormat(data);
			System.out.println("   Clase a predecir movida al ultimo atributo...");
							           	 	
			// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + name + "_" + config);
        	System.out.println("   Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter writer = new BufferedWriter(new FileWriter( bowarff ));
       	 	writer.write(data.toString());
       	 	writer.flush();
       	 	writer.close();
       	 	System.out.println("Conversion " + config + " finalizada... \n");
	    } 
		catch (Exception e) { e.printStackTrace(); }
	}
	
	// FILTRA LOS ATRIBUTOS QUE NO CORRELAN CON LA CLASE Y SI CORRELAN ENTRE SI
	public void filtradoAtributos( String pBow , String pSparse ) 
	{		
		String bow = pBow;
		String sparse = pSparse;
		
		String train_original = this.ruta + "/train_" + bow + "_" + sparse + ".arff" ;
		String test_original = this.ruta + "/test_unk_RaW.arff";
		String train_resultado = this.ruta + "/train_" + bow + "_" + sparse + "_Filtrados.arff";
		String test_resultado = this.ruta + "/test_unk_" + bow + "_" + sparse + "_Filtrados.arff";
		
		try
		{						
			// FILTRAMOS LOS DATOS DE ENTRENAMIENTO
			System.out.println("Iniciando el filtrado de los datos de entrenamiento " + pBow + "_" + pSparse + "...");
			Instances dataTrain = new DataSource( train_original ).getDataSet();
			dataTrain.setClassIndex(0);
			
			// APLICAMOS LA SELECCION DE ATRIBUTOS AL TRAIN EN FUNCION DE LA CORRELACION	        
	        AttributeSelection filtro = new AttributeSelection();
			CfsSubsetEval cfs = new CfsSubsetEval();
			BestFirst search = new BestFirst();
				
			filtro.setSearch(search);
			filtro.setEvaluator(cfs);
			filtro.setInputFormat( dataTrain );
			
			dataTrain = filtro.useFilter( dataTrain , filtro);
			System.out.println("   Datos de entrenamiento filtrados...");
			
			// SETTEAR NOMBRE DE LA RELACION
			dataTrain.setRelationName("preguntas_" + bow + "_" + sparse + "_Filtrados");
        	System.out.println("   Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter trainWriter = new BufferedWriter( new FileWriter( train_resultado ) );
       	 	trainWriter.write( dataTrain.toString() );
       	 	trainWriter.flush();
       	 	trainWriter.close();
       	 	System.out.println("   Filtrado finalizado...");
       	 	
		}
		catch(Exception e) { e.printStackTrace(); }
	}

}
