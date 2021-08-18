<template>
	<v-btn id="post-btn" class="ma-2" rounded outlined color="#00d5aa"
	@click="postDraft">
		글 올리기
	</v-btn>
</template>

<script lang="ts">
	import { Component, Vue } from 'vue-property-decorator';
	import { namespace } from 'vuex-class';
	import http from '@/http/http-common'

	const WriteStoreModule = namespace('writeStore')

	@Component
	export default class PostBtn extends Vue {

		@WriteStoreModule.State('itemList')
		private itemList!:any[]

		// @WriteStoreModule.State('memoList')
		// private memoList!:any[]

		@WriteStoreModule.State('tagList')
		private tagList!:string[]

		@WriteStoreModule.State('titleImage')
		private titleImage!:File

		async postDraft():Promise<void>{
			if(!confirm("발행 하시겠습니까?")){
				return
			}else{
				const date = new Date()
				const form = new FormData()
				const contentList : any[] = []
				const tagList :string[] = []
				//const memoList : any[] = []

				const firstContent = {no:this.itemList[0].id,title:'',subtitle:'',content:this.itemList[0].text};
				contentList.push(firstContent);
				if(this.itemList[0].children.length !== 0){
					for(var i=0;i<this.itemList[0].children.length;i++){
						contentList.push({no:this.itemList[0].children[i].id,
										title: this.itemList[0].children[i].title,
										subtitle: this.itemList[0].children[i].subtitle,
										content: this.itemList[0].children[i].text})
					}
				}

				// for(var j=0;j<this.memoList.length;j++){
                //     memoList.push({no:this.memoList[j].no,memo:this.memoList[j].text})
                // }

				this.tagList.forEach((item) => tagList.push(item))

				const board = JSON.stringify(
						{
							title:this.itemList[0].title,
							subtitle:this.itemList[0].subtitle,
							userEmail:this.$store.state.user.userAccount.attributes.email,
							nickName:this.$store.state.user.userAccount.attributes.nickname,
							titleImageFile:this.titleImage,
							contents:contentList,
							modDatetime:date,
							tags:tagList
						}
					)


				const blob = new Blob([board], {
					type: 'application/json'
				})

				form.append('board', blob)
				form.append('titleImageFile',this.titleImage)

				await http.
					post('/board',form,{
						headers:{
							'Content-Type': "multipart/form-data",
							'Authorization': 'Bearer '+localStorage.getItem('accessToken')
						}
					})
					.then(response => {
						if (response.status >=200 && response.status < 204){
							console.log("Draft post success!")
							//console.log(response.data)
							location.href = '/myonego'
						} else{
							console.log(response)
							console.log("Draft post fail..")
						}
					})
			}
		}
	}
</script>

<style>
#post-btn:hover{
	color:#02bf99 !important;

}
</style>