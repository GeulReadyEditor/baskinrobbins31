<template>
	<v-btn id="saveit-btn" class="ma-2" rounded outlined color="grey"
	@click="saveDraft">
		저장
	</v-btn>
</template>

<script lang="ts">
	import { Component, Vue } from 'vue-property-decorator';
	import { namespace } from 'vuex-class';
	import http from '@/http/http-common'

	const WriteStoreModule = namespace('writeStore')

	@Component
	export default class SaveBtn extends Vue {

		@WriteStoreModule.State('itemList')
		private itemList!:any[]

		@WriteStoreModule.State('memoList')
		private memoList!:any[]

		@WriteStoreModule.State('tagList')
		private tagList!:string[]

		@WriteStoreModule.State('titleImage')
		private titleImage!:File

		@WriteStoreModule.State('imageUrl')
		private imageUrl!:string

		@WriteStoreModule.Getter('getId')
		private id!:string

		async saveDraft():Promise<void>{
			if(!confirm("저장 하시겠습니까?")){
				return
			}else{
				const date = new Date()
				const form = new FormData()
				const contentList : any[] = []
				const memoList : any[] = []
				const tagList :string[] = []

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

                for(var j=0;j<this.memoList.length;j++){
                    memoList.push({no:this.memoList[j].no,memo:this.memoList[j].text})
                }

				this.tagList.forEach((item) => tagList.push(item))

				var tempBoard = ''

				if(this.id !== ''){
					tempBoard = JSON.stringify(
						{
							id:this.id,
							title:this.itemList[0].title,
							subtitle:this.itemList[0].subtitle,
							userEmail:this.$store.state.user.userAccount.attributes.email,
							titleImageFile:this.titleImage,
							contents:contentList,
							memos:memoList,
							modDatetime:date,
							tags:tagList
						}
					)
				}else{
					tempBoard = JSON.stringify(
						{
							title:this.itemList[0].title,
							subtitle:this.itemList[0].subtitle,
							userEmail:this.$store.state.user.userAccount.attributes.email,
							titleImageFile:this.titleImage,
							contents:contentList,
							memos:memoList,
							modDatetime:date,
							tags:tagList
						}
					)
				}


				const blob = new Blob([tempBoard], {
					type: 'application/json'
				})

				form.append('tempBoardDTO', blob)
				form.append('titleImageFile',this.titleImage)

				await http.
					post('/tempBoard',form,{
						headers:{
							'Content-Type': "multipart/form-data",
							'Authorization': 'Bearer '+localStorage.getItem('accessToken')
						}
					})
					.then(response => {
						if (response.status >=200 && response.status < 204){
							console.log("Draft save success!")
							location.href="/myonego"
						} else{
							//console.log(response)
							console.log("Draft save fail..")
						}
					})
			}
		}
	}
</script>

<style>
#saveit-btn:hover{
	color:#757575 !important;
}
</style>