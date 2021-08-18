import { Module, Mutation, VuexModule } from 'vuex-module-decorators';
import store from '@/store/index'

interface ItemList{
	id:number;
	title:string;
	subtitle:string;
	text:string;
	children:Array<{id:number,title:string,subtitle:string,isContent:boolean,text:string}>;
}

interface ContentList{
	no:number;
	title:string;
	subtitle:string;
	content:string;
}
interface MemoList{
	text:string;
	no:number;
}

@Module({store, name:'writeStore', namespaced:true})
export default class writeStore extends VuexModule{
	public tempBoardId = ''
	public titleImage : File|string = new File([""], "")
	public modDatetime : Date = new Date()
	public itemList : ItemList[] = [
		{
			id:1,
			title: '',
			subtitle: '',
			text: '',
			children: Array<{id:number,title:string,subtitle:string,isContent:boolean,text:string}>(),
		}
	]
	public memoList : MemoList[] = []
	public tagList : string[] = []

	public activeVal = 0 //used to communicate between Write.vue and Treeview.vue
	public content = '' //used to communicate between Write.vue and Treeview.vue
	public title = '' //used to communicate between Write.vue and Treeview.vue
	public subtitle = '' //used to communicate between Write.vue and Treeview.vue
	public activeId = 0 //used to communicate between Write.vue and Treeview.vue
	public wordCount = 0 //used to communicate between Write.vue and Tag.vue
	public imageUrl = '' //used to communicate between Tag.vue and PreviewModal.vue
	public isDrag = false

	get getId(){
		return this.tempBoardId
	}

	get getTitleImage(){
		return this.titleImage
	}

	get getActiveVal(){
		return this.activeVal
	}

	get getItemList(){
		return this.itemList
	}

	get getMemoList(){
		return this.memoList
	}

	get getTagList(){
		return this.tagList
	}

	get getWordCount(){
		return this.wordCount
	}

	get getContent(){
		return this.content
	}

	get getTitle(){
		return this.title
	}

	get getSubtitle(){
		return this.subtitle
	}


	// @Mutation
	// 	public update(){
	// 		console.log(this.tagList)
	// 	}

	@Mutation
		public setId(tempBoardId:string){
			this.tempBoardId = tempBoardId
		}

	@Mutation
		public setDate(date:Date){
			this.modDatetime = date
			//console.log(this.modDatetime)
		}

	@Mutation
		public setImageUrl(url:string){
			this.imageUrl = url
			//console.log(this.imageUrl)
		}

	@Mutation
		public setActiveVal(val:any){
			this.activeVal = val[0]
		}

	@Mutation
		public setMainTitle(title:string){
			this.itemList[0].title = title
			//console.log(this.itemList[0].title)
		}

	@Mutation
		public setMainSubtitle(subtitle:string){
			this.itemList[0].subtitle = subtitle
		}

	@Mutation
		public setItemList(list:ContentList[]){
			//console.log(list)
			if(list[0].title===''){
				this.itemList[0].text = list[0].content
				if(list.length>1){
					for(let i=1;i<list.length;i++){
						this.itemList[0].children.push({id:list[i].no,
											title: list[i].title,
											subtitle: list[i].subtitle,
											text: list[i].content,
											isContent: true})
					}
				}
			}else{
				for(let i=0;i<list.length;i++){
					this.itemList[0].children.push({id:list[i].no,
									title: list[i].title,
									subtitle: list[i].subtitle,
									text: list[i].content,
									isContent: true})
				}
			}
		}

	@Mutation
		public setTagList(list:string[]){
			this.tagList = list
		}

	@Mutation
		public setMemoList(list:any[]){
			//console.log(list)
			for(let i=0;i<list.length;i++){
				this.memoList.push({no:list[i].no,text:list[i].memo})
			}
			//console.log(this.memoList[0].text)
		}

	@Mutation
		public setTitleImage(image:File){
			this.titleImage = image
			//console.log(this.titleImage)
		}

	@Mutation
		public sendTreeTitle(title:string){
			//console.log(title)
			if(this.itemList[0].id == this.activeVal){
				this.title = title
				//console.log(this.content)
			}
			for(let i=0; i<this.itemList[0].children.length; i++){
				if(this.itemList[0].children[i].id == this.activeVal){
					this.title = title
					//console.log(this.content)
				}
			}
		}

	@Mutation
		public sendContent(itemId:number){ //triggers when item in Treeview.vue is clicked
			//console.log(this.itemList[0].title)
			this.activeId = itemId
			if(this.itemList[0].id == itemId){
				this.title = this.itemList[0].title
				this.subtitle = this.itemList[0].subtitle
				const content:string ='<p>'+this.itemList[0].text.replace('\n','</p><p>')+'</p>'
				this.content = content
			}// displays title + text on text editor
			for(let i=0; i<this.itemList[0].children.length; i++){
				if(this.itemList[0].children[i].id == itemId){
					this.title = this.itemList[0].children[i].title
					this.subtitle = this.itemList[0].children[i].subtitle
					const content:string = '<p>'+this.itemList[0].children[i].text.replace('\n','</p><p>')+'</p>'
					this.content = content
				}// displays subtitle + text on text editor
			}
			//console.log(itemId)
		}

	@Mutation
		public setContent(data:string){
			this.content = data
		}

	@Mutation
		public updateContent(data:string){ //triggers when key is pressed in Write.vue
			//console.log(this.activeId)
			//console.log(data)
			const str = data
			str.replaceAll("<h1>","<p>")
			str.replaceAll("</h1>","</p>")
			str.replaceAll("<h2>","<p>")
			str.replaceAll("</h2>","</p>")
			str.replaceAll("<h3>","<p>")
			str.replaceAll("</h3>","</p>")
			const content = str.substring(str.indexOf("<p>")+3,str.lastIndexOf("</p>"))

			if(this.activeId == 1){
				this.itemList[0].text = content.replace('</p><p>','\n')
			}
			if(this.activeId > 1){
				for(let i=0; i<this.itemList[0].children.length; i++){
					if(this.itemList[0].children[i].id == this.activeId){
						this.itemList[0].children[i].text = content.replace('</p><p>','\n')
					}
				}
			}
			this.wordCount = content.length
		}

	@Mutation
		public setTitle(data:string){
			this.title = data
		}

	@Mutation
		public updateTitle(data:string){ //triggers when key is pressed in Write.vue
			//console.log(this.activeId)
			//console.log(data)

			if(this.activeId == 1){
				this.itemList[0].title = data
			}
			if(this.activeId > 1){
				for(let i=0; i<this.itemList[0].children.length; i++){
					if(this.itemList[0].children[i].id == this.activeId){
						this.itemList[0].children[i].title = data
					}
				}
			}
		}

	@Mutation
		public setSubtitle(data:string){
			this.subtitle = data
		}

	@Mutation
		public updateSubtitle(data:string){ //triggers when key is pressed in Write.vue
			//console.log(this.activeId)
			//console.log(data)

			if(this.activeId == 1){
				this.itemList[0].subtitle = data
			}
			if(this.activeId > 1){
				for(let i=0; i<this.itemList[0].children.length; i++){
					if(this.itemList[0].children[i].id == this.activeId){
						this.itemList[0].children[i].subtitle = data
					}
				}
			}
		}

	@Mutation
		public deleteMemo(index:number){
			this.memoList.splice(index, 1)
		}
	@Mutation
		public dragMemoList(val:any){
			this.memoList = val
			//console.log("hello")
			this.isDrag = true
		}

  }