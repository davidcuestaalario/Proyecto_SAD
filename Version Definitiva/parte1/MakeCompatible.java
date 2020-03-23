package parte1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MakeCompatible 
{
	private String ruta;
	
	public MakeCompatible( String pRuta ) { this.ruta = pRuta; }
	
	// CONVIERTE EL FICHERO DE TEST AL MISMO FORMATO QUE EL DE TRAIN UTILIZANDO REMOVE
	public void RemoveTest( String pBow , String pSparse ) 
	{
		String train_filtrado = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String test_original = this.ruta + "/test_unk_" + pBow + "_" + pSparse + ".arff";
		String test_resultado = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados.arff";
		
		try 
		{
			System.out.println("Iniciando la compatibilizacion de los datos de test " + pBow + "_" + pSparse + "...");
			
			BufferedReader encabezado = new BufferedReader(new FileReader( train_filtrado ));
	    	BufferedReader cuerpo = new BufferedReader(new FileReader( test_original ));
	    	BufferedWriter write = new BufferedWriter(new FileWriter( test_resultado ));
				
	    	DataSource trainSource = new DataSource( train_filtrado );
	    	DataSource testSource = new DataSource( test_original );
	    	Instances train = trainSource.getDataSet();
	    	Instances test = testSource.getDataSet();
		   	if (train.classIndex() == -1) { train.setClassIndex(train.numAttributes() - 1); }
		   	if (test.classIndex() == -1) { test.setClassIndex(test.numAttributes() - 1); }
		    
		    int[] indicesANoEliminar = new int[train.numAttributes()-1];
			
			ArrayList<Attribute> atributosPorDefecto = new ArrayList<Attribute>();
			Enumeration<Attribute> at = train.enumerateAttributes();
			while(at.hasMoreElements()) 
			{
				atributosPorDefecto.add(at.nextElement());
			}
				
			int i=0;
			Enumeration<Attribute> at2 = train.enumerateAttributes();
			while(at2.hasMoreElements()) 
			{
				Attribute atributo = at2.nextElement();
				if(atributosPorDefecto.contains(atributo)) 
				{
					indicesANoEliminar[i] = atributosPorDefecto.indexOf(atributo);
					i++;
				}				
			}
			
			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesANoEliminar);
			remove.setInvertSelection(true);
			remove.setInputFormat(train);
			test = Filter.useFilter(test, remove);
			    
		    // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
		    write.write( test.toString() );
		    write.flush();
		    write.close();
		    System.out.println("   Filtrado finalizado... \n");
		    
			} 
			catch(Exception e) { e.printStackTrace(); }			 
		}

	// AJUSTA LAS CABECERAS DEL TEST PARA ANADIR LA CLASE
	public void compatibleTest( String pBow , String pSparse ) 
	{
		boolean sparse = true;
		if( pSparse == "NonSparse" ) { sparse = false; }
    	boolean bow = true;
    	if( pBow == "TFIDF" ) { bow = false; }

		String test_original = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados.arff";
		String test_resultado = this.ruta + "/test_unk_" + pBow + "_" + pSparse + "_Filtrados_Ajustados.arff";
		String train_original = this.ruta + "/train_" + pBow + "_" + pSparse + "_Filtrados.arff";
			
		try 
		{
			System.out.println("Inicindo la conversion de los datos del test para el formato " + pBow + "_" + pSparse + "...");
			
	    	BufferedReader encabezado = new BufferedReader(new FileReader( train_original ));
	    	BufferedReader cuerpo = new BufferedReader(new FileReader( test_original ));
	    	BufferedWriter write = new BufferedWriter(new FileWriter( test_resultado ));
				
	    	String word = "@data";
		    	
	    	// COPIAMOS EL ENCABEZADO DEL TRAIN
	    	String lineCabeza;
	    	boolean fin = false;
	    	while( (lineCabeza = encabezado.readLine()) != null )
	    	{
	    		if ( !fin ) 
	    		{
	    			//System.out.println( lineCabeza.contains( word ) );
		    		//System.out.println(lineCabeza);
	    			write.write( lineCabeza );
		    		write.newLine();
	    		}
	    		if( lineCabeza.contains( word ) ) { fin = true; }
	    	}
		    	
	    	// COPIAMOS LOS DATOS DEL TEST
	    	String lineCuerpo;
	    	while( (lineCuerpo = cuerpo.readLine()) != null )
	    	{
	    		if ( !fin ) 
	    		{
	    			//System.out.println( lineCabeza.contains( word ) );
		    		//System.out.println(lineCabeza);
	    			write.write( lineCuerpo );
		    		write.newLine();
	    		}
	    		if( lineCuerpo.contains( word ) ) { fin = false; }
	    	}
	    	write.close();
		}
		catch(Exception e) { e.printStackTrace(); }
			
		}
}
