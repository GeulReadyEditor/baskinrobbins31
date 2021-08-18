<template>
	<v-btn id="delete-btn" class="ma-2" rounded outlined color="grey"
	@click="deleteDraft">
		삭제
	</v-btn>
</template>

<script lang="ts">
	import { Component, Vue } from 'vue-property-decorator';
	import { namespace } from 'vuex-class';
	import http from '@/http/http-common'

	const WriteStoreModule = namespace('writeStore')

	@Component
	export default class DeleteBtn extends Vue {

		@WriteStoreModule.Getter('getId')
		private id!:string

		async deleteDraft():Promise<void>{
			if(!confirm("삭제 하시겠습니까?")){
				return
			}else{
				await http.
					delete('/tempBoard',{
						params:{
							'boardIdx': this.id
						},
						headers:{
							'Authorization': 'Bearer '+localStorage.getItem('accessToken')
						}
					})
					.then(response => {
						if (response.status >=200 && response.status < 204){
							console.log("Draft delete success!")
							location.href="/myonego"
						} else{
							//console.log(response)
							console.log("Draft delete fail..")
						}
					})
			}
		}
	}
</script>

<style>
#delete-btn:hover{
	color:#757575 !important;
}
</style>