<template>
  <div style="margin-bottom:-400px;">
	<div style="margin-top:-36px;"><v-divider/></div>
	<!-- 목록 -->
	<treeview :tempBoardId="tempBoardId"/>

	<div align="center" id="write-margin" >
		<div id="editor-size">

		  <div style="width:100%;float:left;">
			  <v-card class="title-card" height="6.1vh" elevation="2" color="#f5f5f5">
				<!-- 제목 -->
				<div style="width:50%;height:6.1vh;padding-top:4px;padding-bottom:4px;float:left;">
				  <v-text-field class="mx-2"
				    v-model="title"
				  	height="5.1vh"
					color="#00d5aa"
					background-color="white"
					:placeholder="titlePlace"
					:label="titleLabel"
					dense outlined/>
				</div>
				<!-- 소제목 -->
				<div style="width:50%;height:6.1vh;padding-top:4px;padding-bottom:4px;float:left;">
				  <v-text-field class="mx-2"
				    v-model="subtitle"
				  	height="5.1vh"
					color="#00d5aa"
					background-color="white"
					:placeholder="subtitlePlace"
					:label="subtitleLabel"
					dense outlined>
					<template v-slot:append-outer v-if="activeVal === 1">
					  <v-hover v-slot="{ hover }">
						<v-tooltip top>
							<template v-slot:activator="{ on, attrs }">
								<v-btn icon :color="hover ? '#00d5aa' : ''" v-bind="attrs" v-on="on" @click="generateSub"><v-icon>mdi-clipboard-text-play-outline</v-icon></v-btn>
							</template>
							<span>소제목 자동생성</span>
				        </v-tooltip>
					  </v-hover>
					</template>
				  </v-text-field>
				</div>
			  </v-card>
		  </div>
		  <!-- 텍스트 에디터 -->
		  <div style="width:100%;float:left;">
			<tiptap-vuetify id="text-editor"
				v-model="content"
				:extensions="extensions"
				:native-extensions="nativeExtensions"
				:disabled="isActive"
				autoFocus=true
				placeholder="글 내용을 입력하세요"
				>
			</tiptap-vuetify>
		  </div>
		  <!-- 태그 입력, 이미지 선택, 글자수 -->
		  <tag />
		</div>
	</div>
	<!-- 자동완성 & 메모 -->
	<div id="memo-margin">
		<div id="memo-size">
		  <!-- 자동완성 -->
		  <autocom />
		  <v-divider style="margin-bottom:7px;"/>
		  <!-- 메모 -->
		  <memo />
		</div>
	</div>
  </div>
</template>

<script lang="ts">
	import { Component, Prop, Vue, Watch } from 'vue-property-decorator';
	// eslint-disable-next-line @typescript-eslint/ban-ts-comment
	// @ts-ignore
	import { TiptapVuetify, History, Paragraph, Heading, Bold, Italic, Underline } from "tiptap-vuetify";
	// eslint-disable-next-line @typescript-eslint/ban-ts-comment
	// @ts-ignore
	import { insertText } from 'tiptap-commands';
	// eslint-disable-next-line @typescript-eslint/ban-ts-comment
	// @ts-ignore
	import { Extension } from 'tiptap'
	import Treeview from '@/components/layout/write/Treeview.vue'
	import Autocom from '@/components/layout/write/Autocom.vue'
	import Memo from '@/components/layout/write/Memo.vue'
	import Tag from '@/components/layout/write/Tag.vue'
	import '@/assets/css/Write.css'
	import { namespace } from 'vuex-class';
	import http from '@/http/http-common'
	import { eventBus } from '@/main'

	const WriteStoreModule = namespace('writeStore')

	@Component({
		components:{
			TiptapVuetify,
			'treeview':Treeview,
			'autocom':Autocom,
			'memo':Memo,
			'tag':Tag
		}

	})
	export default class Write extends Vue {
		@Prop({ type: String }) readonly tempBoardId!: string
		@Prop({ type: String }) readonly boardId!: string

		extensions : unknown = [
			History,
			Paragraph,
			[Heading, { options: {levels:[1,2,3]}}],
			Bold,
			Italic,
			Underline
		]
		nativeExtensions : unknown = [
			new class extends Extension {
				keys() {
    				return {
      					Tab: insertText('    ')
    				}
 				}
			}
		]
		titleLabel = '글 제목'
		subtitleLabel = '글 소제목'
		titlePlace = '글 제목을 입력하세요'
		subtitlePlace = '글 소제목을 입력하세요'
		title = ''
		subtitle = ''
		errored = false
		loading = true
		content = ''
		isActive = false

		@Watch('activeVal')
		watchActiveVal():void{
			//console.log(this.activeVal)
			if(this.activeVal == undefined){
				this.isActive = true
			}else this.isActive = false

			if(this.activeVal > 1){
				this.titleLabel = '목차 이름'
				this.subtitleLabel = '목차 소제목'
				this.titlePlace = '목차 이름을 입력하세요'
				this.subtitlePlace = '목차 소제목을 입력하세요'
			}else if(this.activeVal === 1){
				this.titleLabel = '글 제목'
				this.subtitleLabel = '글 소제목'
				this.titlePlace = '글 제목을 입력하세요'
				this.subtitlePlace = '글 소제목을 입력하세요'
			}
		}

		@Watch('getContent')
		watchGetContent():void{
			this.content = this.getContent
			this.updateContent(this.content)
		}

		@Watch('content')
		watchContent():void{
			//console.log(this.content)
			this.setContent(this.content)
			this.updateContent(this.content)
		}

		@Watch('getTitle')
		watchGetTitle():void{
			this.title = this.getTitle
			this.updateTitle(this.title)
		}

		@Watch('title')
		watchTitle():void{
			this.setTitle(this.title)
			this.updateTitle(this.title)
		}

		@Watch('getSubtitle')
		watchGetSubtitle():void{
			this.subtitle = this.getSubtitle
			this.updateSubtitle(this.subtitle)
		}

		@Watch('subtitle')
		watchSubtitle():void{
			this.setSubtitle(this.subtitle)
			this.updateSubtitle(this.subtitle)
		}

		@WriteStoreModule.Getter('getId') private getId!:string

		@WriteStoreModule.Getter('getContent') private getContent!:string

		@WriteStoreModule.Mutation('setContent') private setContent!:(data:string)=>void

		@WriteStoreModule.Mutation('updateContent') private updateContent!:(data:string)=>void

		@WriteStoreModule.Getter('getTitle') private getTitle!:string

		@WriteStoreModule.Mutation('setTitle') private setTitle!:(data:string)=>void

		@WriteStoreModule.Mutation('updateTitle') private updateTitle!:(data:string)=>void

		@WriteStoreModule.Getter('getSubtitle') private getSubtitle!:string

		@WriteStoreModule.Mutation('setSubtitle') private setSubtitle!:(data:string)=>void

		@WriteStoreModule.Mutation('updateSubtitle') private updateSubtitle!:(data:string)=>void

		@WriteStoreModule.Getter('getActiveVal') private activeVal!:number

		@WriteStoreModule.Mutation('setId') private setId!:(tempBoardId:string)=>void

		@WriteStoreModule.Mutation('setMemoList') private setMemoList!:(list:any[])=>void

		@WriteStoreModule.Mutation('setTagList') private setTagList!:(list:string[])=>void

		@WriteStoreModule.Mutation('setItemList') private setItemList!:(list:any[])=>void

		@WriteStoreModule.Mutation('setMainTitle') private setMainTitle!:(title:string)=>void

		@WriteStoreModule.Mutation('setMainSubtitle') private setMainSubtitle!:(subtitle:string)=>void

		@WriteStoreModule.Mutation('setImageUrl') private setImageUrl!:(url:string)=>void

		@WriteStoreModule.Getter('getItemList') private itemList!:any[]

		created(){
			history.pushState(null, '', document.URL)
			window.onpopstate = () => {
				this.confirmBack()
				history.pushState(null, '', document.URL)
			}
			if(this.tempBoardId !== undefined) this.editTempBoard(this.tempBoardId)
			if(this.boardId !== undefined) this.editBoard(this.boardId)
			//this.content = this.getContent
			//console.log(this.getId)
		}

		confirmBack():void{
			if(!confirm('글 저장이 안됐습니다. 이동하시겠습니까?')){
				return
			}else{
				location.href = '/myonego'
			}
		}

		async editTempBoard(tempBoardId:string):Promise<void>{
			//console.log(tempBoardId)
			await http.
				get('/tempBoard',{
					params:{
						'tempBoardId': tempBoardId
					},
					headers:{
						'Authorization': 'Bearer '+localStorage.getItem('accessToken')
					}
				})
				.then(response => {
					//console.log(response.data[0].id)
					this.setId(response.data[0].id)
					this.setMainTitle(response.data[0].title)
					this.setMainSubtitle(response.data[0].subtitle)
					this.setItemList(response.data[0].contents)
					this.setTagList(response.data[0].tags)
					this.setMemoList(response.data[0].memos)
					this.setImageUrl(response.data[0].titleImage)
					//console.log(this.getId)
					eventBus.$emit("clickFirstTree")
				})
				.catch(() => this.errored = true )
				.finally(() => {
					this.loading = false
				})
		}

		async editBoard(boardId:string):Promise<void>{
			//console.log(tempBoardId)
			await http.
				get('/board',{
					params:{
						'boardId': boardId
					},
					headers:{
						'Authorization': 'Bearer '+localStorage.getItem('accessToken')
					}
				})
				.then(response => {
					console.log(response.data)
					this.setId(response.data[0].id)
					this.setMainTitle(response.data[0].title)
					this.setMainSubtitle(response.data[0].subtitle)
					this.setItemList(response.data[0].contents)
					this.setTagList(response.data[0].tags)
					// this.setMemoList(response.data[0].memos)
					this.setImageUrl(response.data[0].titleImage)
					//console.log(this.getId)
					eventBus.$emit("clickFirstTree")
				})
				.catch(() => this.errored = true )
				.finally(() => {
					this.loading = false
				})
		}

		generateSub():void{
			var subtitle = ''

			var contentList = [] as string[]
			var contents = [] as string[]
			contentList.push(this.itemList[0].text.replaceAll("\n"," ").replaceAll("</p><p>"," "))
			if(this.itemList[0].children.length !== 0){
				for(var i=0; i<this.itemList[0].children.length;i++){
					contentList.push(this.itemList[0].children[i].text.replaceAll("\n"," ").replaceAll("</p><p>"," "))
				}
			}
			contents.push(contentList.join(" "))
			console.log(contents)

			////////////////axios here///////////////////

			//this.subtitle = subtitle
		}

	}
</script>

<style>
/* .v-application .write-header-color .white{
	background-color:#282828 !important;
} */
</style>